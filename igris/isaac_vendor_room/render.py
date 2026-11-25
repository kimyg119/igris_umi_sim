"""Render authoritative vendor body poses in Isaac Sim; never step Isaac physics.

--live follows latest.bin; default replays the recorded native poses.bin.
RGB/depth are Isaac sensor outputs; proprioception/contact stay vendor-native.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from pose_frames import frames_at

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--scene',type=Path,help='Relocated recorded USD; leaves historical manifest unchanged')
parser.add_argument('--gui',action='store_true')
parser.add_argument('--live',action='store_true')
parser.add_argument('--seconds',type=float,default=6.)
parser.add_argument('--start-wall',type=float)
parser.add_argument('--start-sim',type=float,help='Replay using native simulation timestamps from this time')
parser.add_argument('--playback-speed',type=float,default=1.,help='Source seconds per video second')
parser.add_argument('--camera',action='append',help='One camera path; repeat for several. Default: three synchronized views')
parser.add_argument('--capture',type=Path)
parser.add_argument('--record',action='store_true',help='Save every live frame; replay always saves frames')
parser.add_argument('--fps',type=float,default=15.,help='Output video frames per second')
# The renderer builds each image from several of its own frames.  Bodies jump to the next pose between
# saved frames, so with too few of them the moving parts are a blend of old and new positions.
parser.add_argument('--subframes',type=int,default=2,help='Renderer frames accumulated per saved frame; 16 keeps moving parts clean')
parser.add_argument('--antialiasing',type=int,default=3,choices=range(5),help='0 off, 1 TAA, 2 FXAA, 3 DLSS (upscaled), 4 DLAA (native resolution)')
parser.add_argument('--interpolate',action='store_true',help='Replay: pose at each frame time between the recorded rows, not the next recorded row')
args,extra=parser.parse_known_args()
if args.subframes<1:parser.error('--subframes must be at least 1')
args.out=args.out.resolve()
manifest=json.loads((args.out/'manifest.json').read_text())
meta=json.loads((args.out/'model.json').read_text())
runtime=args.out/'runtime.json'
if runtime.exists():
    cpu=json.loads(runtime.read_text()).get('physics_cpu')
    if cpu is not None:
        sibling=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
        avoid={cpu}
        if sibling.exists():
            for part in sibling.read_text().strip().split(','):
                limits=list(map(int,part.split('-')));avoid.update(range(limits[0],limits[-1]+1))
        allowed=os.sched_getaffinity(0)-avoid
        if allowed:os.sched_setaffinity(0,allowed)
width=2+len(meta['bodies'])*7
capture=(args.capture or args.out/('live_render' if args.live else 'render')).resolve()
capture.mkdir(parents=True,exist_ok=False)

from isaacsim import SimulationApp
app=SimulationApp({'headless':not args.gui,'width':1280,'height':960,'renderer':'RayTracedLighting',
                   'anti_aliasing':args.antialiasing,'limit_cpu_threads':8,'extra_args':['--/exts/isaacsim.physics.newton/auto_switch_on_startup=false',
                   '--/exts/isaacsim.core.simulation_manager/default_engine=physx']})
import carb
import omni.usd
import omni.timeline
import omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
from PIL import Image

report={'physics_backend':'external vendor MuJoCo 3.2.7','isaac_physics_steps':0,
        'mode':'live' if args.live else 'recorded physics replay','hardware_executed':False,'pass':False}
try:
    # Prepare outside the live USD context: removing physics schemas after
    # Fabric population starts crashes the installed Kit 110 build.
    stage=Usd.Stage.Open(str(args.scene.resolve()) if args.scene else manifest['source_usd'])
    stage.SetEditTarget(stage.GetSessionLayer())
    # Recorded USD layers can contain absolute texture paths from another user.
    # Relocate only this project's assets in the anonymous render session.
    project=Path(__file__).resolve().parents[2]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value=attr.Get()
            if isinstance(value,Sdf.AssetPath) and '/Project_Humanoid/' in value.path:
                path=project/value.path.split('/Project_Humanoid/',1)[1]
                if not path.is_file():raise FileNotFoundError(path)
                attr.Set(Sdf.AssetPath(str(path)))
    timeline=omni.timeline.get_timeline_interface();timeline.stop()
    cache=UsdGeom.XformCache()
    names={b['name']:i for i,b in enumerate(meta['bodies'])}
    fixed={};articulated={}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            joint=UsdPhysics.Joint(prim);a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
            if a and b:fixed[str(b[0])]=str(a[0])
        if prim.IsA(UsdPhysics.Joint) and manifest.get('compliant_hands'):
            for side in ('l','r'):
                for part in ('mcp','dip'):
                    if prim.GetName()==f'{side}_hand_dof_joint_{part}':
                        targets=UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                        if targets:articulated[str(targets[0])]=names[f'{side}_hand_{part}']
    if manifest.get('compliant_hands') and len(articulated)!=4:raise ValueError('Missing USD MCP/DIP bindings')
    room={b['usd_path']:names[b['body']] for b in manifest['room_bodies']}
    bindings=[]
    robot_count=0
    # Resolve every robot link, including fixed hand and camera sub-links.
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        if path.startswith('/World/IGRIS_C'):
            anchor=path;seen=set()
            while anchor not in articulated and anchor.rsplit('/',1)[-1] not in names:
                if anchor in seen or anchor not in fixed:raise ValueError(f'Unmapped robot body: {path}')
                seen.add(anchor);anchor=fixed[anchor]
            index=articulated[anchor] if anchor in articulated else names[anchor.rsplit('/',1)[-1]]
            offset=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(stage.GetPrimAtPath(anchor)).GetInverse()
            robot_count+=1
        elif path in room:
            index=room[path];offset=Gf.Matrix4d(1.)
        else:raise ValueError(f'Unmapped dynamic body: {path}')
        xform=UsdGeom.Xformable(prim)
        xform.MakeMatrixXform();xform.SetResetXformStack(False)
        parent_inverse=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        bindings.append((index,offset,path,parent_inverse))
    # Disable simulation schemas only in the anonymous session layer. Even if the
    # user presses Play, no second engine can take ownership of rendered bodies.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):prim.SetActive(False)
        elif prim.IsA(UsdPhysics.Scene):prim.SetActive(False)
        else:
            for api in (UsdPhysics.RigidBodyAPI,UsdPhysics.CollisionAPI,UsdPhysics.ArticulationRootAPI):
                if prim.HasAPI(api):prim.RemoveAPI(api)
    stage.Flatten().Export(str(capture/'render_scene.usdc'))
    print('RENDER_STAGE_PREPARED',flush=True)
    omni.usd.get_context().open_stage(str(capture/'render_scene.usdc'))
    for _ in range(12):app.update()
    print('RENDER_STAGE_OPENED',flush=True)
    stage=omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    bindings=[(index,offset,UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0],parent_inverse) for index,offset,path,parent_inverse in bindings]
    report.update(mapped_robot_links=robot_count,mapped_room_bodies=len(room),articulated_hand_anchors=articulated)
    settings=carb.settings.get_settings();settings.set('/rtx/post/tonemap/op',4)
    settings.set('/rtx/post/tonemap/filmIso',320.)
    settings.set('/rtx/post/tonemap/whitepoint',6500.)
    # What the renderer actually runs with: this Isaac Sim answers RealTimePathTracing where RayTracedLighting was asked for.
    report.update(render_mode=settings.get('/rtx/rendermode'),antialiasing=settings.get('/rtx/post/aa/op'),subframes=args.subframes)
    from omni.kit.viewport.utility import get_active_viewport
    viewport=get_active_viewport()
    cameras=args.camera or ['/World/Cameras/Overview','/World/Cameras/RobotFacing','/World/Cameras/RolltainerDown']
    # Front is +X in the native robot frame. The third eye is over the cart,
    # looking down toward the robot; only this rendering session is edited.
    for name,eye,target,focal in [
        ('RobotFacing',(2.0,-.8,1.5),(0.,0.,.86),30.),
        ('RolltainerDown',(.35,1.25,2.7),(0.,0.,.85),32.),
        ('TransportWide',(1.9,-.75,1.8),(0.,.65,.85),24.),
        ('PlacementDetail',(-.65,1.45,2.1),(.0283,1.396,.78),30.),
        ('NearPlacementDetail',(1.1,.6,1.55),(.244,.237,.80),30.),
        ('ForwardTransportWide',(1.7,1.65,1.8),(.55,-.10,.80),24.),
        ('ForwardPlacementDetail',(1.9,.35,1.5),(1.083,-.261,.80),30.),
        ('ForwardPlacementSide',(.10,-.80,1.40),(1.083,-.261,.80),30.),
    ]:
        camera=UsdGeom.Camera.Define(stage,'/World/Cameras/'+name)
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix)
        camera.CreateFocalLengthAttr(focal);camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
    if len(set(cameras))!=len(cameras):raise ValueError('Duplicate camera path')
    if viewport:viewport.set_active_camera(cameras[0])
    sensors=[]
    for path in cameras:
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):raise ValueError(f'Missing camera: {path}')
        product=rep.create.render_product(path,(1280,960))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth.attach([product])
        directory=capture/path.rsplit('/',1)[-1] if len(cameras)>1 else capture
        directory.mkdir(exist_ok=True);(directory/'frames').mkdir()
        sensors.append((path,rgb,depth,directory))
    report['cameras']={path:{} for path in cameras}
    print('RENDER_SENSORS_CREATED',flush=True)
    def update(row):
        if row.shape!=(width,) or not np.isfinite(row).all():raise ValueError('Invalid native pose frame')
        poses=row[2:].reshape(-1,7)
        expected_world=[]
        for index,offset,op,parent_inverse in bindings:
            p=poses[index]
            if abs(np.linalg.norm(p[3:])-1)>.001:raise ValueError('Invalid native quaternion')
            matrix=Gf.Matrix4d().SetRotate(Gf.Quatd(float(p[3]),Gf.Vec3d(*p[4:7])))
            matrix.SetTranslateOnly(Gf.Vec3d(*p[:3]))
            desired=offset*matrix;op.Set(desired*parent_inverse);expected_world.append(desired)
        # Check the composed USD world transform, not merely the authored local op.
        check=UsdGeom.XformCache()
        for (_,_,op,_),expected in zip(bindings,expected_world):
            actual=check.GetLocalToWorldTransform(op.GetAttr().GetPrim())
            if not np.allclose(np.asarray(actual),np.asarray(expected),atol=1e-6):
                raise RuntimeError('Rendered world transform differs from native pose')
    rows=None
    if not args.live:
        raw=np.fromfile(args.out/'poses.bin',dtype=np.float64)
        rows=raw[:len(raw)//width*width].reshape(-1,width)
        if not len(rows):raise ValueError('No recorded physics frames')
        if args.start_sim is not None and args.start_wall is not None:raise ValueError('Choose one replay clock')
        column=1 if args.start_sim is not None else 0
        first=args.start_sim if column==1 else args.start_wall if args.start_wall is not None else max(rows[0,0],rows[-1,0]-args.seconds)
        if args.seconds<=0 or not np.isfinite([args.fps,args.playback_speed,first]).all() or args.fps<=0 or args.playback_speed<=0:
            raise ValueError('Replay requires positive duration and frame rate')
        # Native time can reset during bringup. Only the final monotonic epoch
        # can describe the subsequent uninterrupted transport experiment.
        resets=np.flatnonzero(np.diff(rows[:,column])<0)
        if len(resets):rows=rows[resets[-1]+1:]
        if first<rows[0,column] or first>=rows[-1,column]:raise ValueError('Replay start outside final recorded epoch')
        times=np.arange(first,min(first+args.seconds,rows[-1,column]),args.playback_speed/args.fps)
        selected=frames_at(rows,column,times) if args.interpolate else rows[np.minimum(np.searchsorted(rows[:,column],times),len(rows)-1)]
        report.update(replay_clock='native_sim' if column==1 else 'wall',playback_speed=args.playback_speed,interpolated=args.interpolate,
                      requested_source_start=float(first),requested_source_duration_s=args.seconds)
        if len(selected)<2:raise ValueError('No recorded interval selected')
        update(selected[0])
    else:update(np.fromfile(args.out/'latest.bin',dtype=np.float64))
    print('RENDER_FIRST_POSE_APPLIED',flush=True)
    for index in range(40):
        app.update()
        if (index+1)%10==0:print('RENDER_WARMUP',index+1,flush=True)
    stamps=[];frame=0;start=time.monotonic();last_pose=None
    print('ISAAC_VENDOR_RENDER_READY',json.dumps(report),flush=True)
    while app.is_running():
        if (capture/'stop.request').exists():break
        if args.live:
            if args.seconds>0 and time.monotonic()-start>=args.seconds:break
            row=np.fromfile(args.out/'latest.bin',dtype=np.float64)
            age=time.time()-row[0]
            if not 0<=age<1.:raise RuntimeError(f'Native physics stream stale: {age:.3f} s')
            if last_pose==row[0]:app.update();continue
        else:
            if frame>=len(selected):break
            row=selected[frame]
        last_pose=row[0];update(row)
        # Trigger sensor evaluation without advancing Isaac's physics clock.
        rep.orchestrator.step(rt_subframes=args.subframes,delta_time=0.,pause_timeline=True)
        for path,rgb,depth,directory in sensors:
            image=np.asarray(rgb.get_data())[:,:,:3]
            if image.shape!=(960,1280,3):raise RuntimeError(f'RGB sensor returned no image: {path}')
            if frame==0:
                Image.fromarray(image).save(directory/'first.png')
                d=np.asarray(depth.get_data());np.save(directory/'depth_first.npy',d)
                finite=d[np.isfinite(d)&(d>0)]
                report['cameras'][path].update(rgb_mean=float(image.mean()),rgb_std=float(image.std()),
                    valid_depth_fraction=float(finite.size/d.size),depth_min_m=float(finite.min()) if finite.size else None)
            if args.record or not args.live:
                Image.fromarray(image).save(directory/'frames'/f'{frame:05d}.png',compress_level=1)
        stamps.append([float(row[0]),float(row[1]),time.time()]);frame+=1
    if frame:
        for path,rgb,depth,directory in sensors:
            Image.fromarray(np.asarray(rgb.get_data())[:,:,:3]).save(directory/'last.png')
    if len(cameras)==1:report.update(report['cameras'][cameras[0]])
    report.update(frames=frame,source_timestamps=stamps,
                  replay_fps=args.fps if not args.live else None,
                  pass_=bool(frame>1 and all(v['rgb_std']>15 and 20<v['rgb_mean']<230 and v['valid_depth_fraction']>.2 for v in report['cameras'].values())))
    report['pass']=report.pop('pass_')
    if not report['pass']:raise RuntimeError('Rendered RGB/depth validation failed')
    if args.record or not args.live:
        for path,rgb,depth,directory in sensors:
            if args.live:
                intervals=np.diff([s[0] for s in stamps])
                durations=[*intervals,float(np.median(intervals))]
                listing='ffconcat version 1.0\n'+''.join(f"file frames/{i:05d}.png\nduration {dt:.9f}\n" for i,dt in enumerate(durations))
                listing+=f'file frames/{frame-1:05d}.png\n'
                (directory/'frames.ffconcat').write_text(listing)
                inputs=['-f','concat','-safe','0','-i',str(directory/'frames.ffconcat'),'-vsync','vfr']
            else:inputs=['-framerate',str(args.fps),'-i',str(directory/'frames/%05d.png')]
            video=directory/'recording.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n',*inputs,'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(video)],check=True)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
            report['cameras'][path]['video']=str(video)
            report['cameras'][path]['full_video_decode_pass']=True
except Exception as exc:
    report['pass']=False;report['error']=str(exc);raise
finally:
    (capture/'render_report.json').write_text(json.dumps(report,indent=2))
    print('RENDER_RESULT',json.dumps({k:v for k,v in report.items() if k!='source_timestamps'}),flush=True)
    app.close()
