"""Per-original-frame UMI errors and a video of recorded native vendor physics."""
import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
from collections import Counter

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from render_results import model_for
from umi_replay import pose_error
from igris_ik.constants import URDF_JOINT_NAMES
from igris_ik.assets import validate_urdf
from igris_ik.vendor_limits import coupled_halfspaces


def stats(values):
    values=np.asarray(values)
    if not len(values):return None
    return dict(mean=np.mean(values,axis=0).tolist(),median=np.median(values,axis=0).tolist(),
                p95=np.percentile(values,95,axis=0).tolist(),max=np.max(values,axis=0).tolist())


def process(out, video=True):
    meta=json.loads((out/'metadata.json').read_text());target=np.load(out/'targets.npz')
    live=[json.loads(line) for line in (out/'native_frames.jsonl').open()]
    audit=[json.loads(line) for line in (out/'static_ik.jsonl').open()]
    if len(live)!=len(audit) or len(live)!=meta['source_frames']:
        raise RuntimeError('Per-frame audit/replay is incomplete')
    binary=(out/'physics.bin').read_bytes();nq,nv=struct.unpack('ii',binary[:8]);width=2+nq+nv+7
    size=(len(binary)-8)//(8*width)*8*width
    recorded=np.frombuffer(binary[8:8+size],np.float64).reshape(-1,width)
    t=recorded[:,0]-meta['start_wall'];keep=(t>=0)&(t<=meta['duration_s']);recorded=recorded[keep];t=t[keep]
    m=model_for(out);assert m.nq==nq;d=mujoco.MjData(m)
    base=m.body('pelvis').id;hands=[m.body(name).id for name in meta['tools']]
    actual=[];base_positions=[];tilts=[]
    for row in recorded:
        d.qpos[:]=row[2:2+nq];mujoco.mj_forward(m,d)
        bp=d.xpos[base].copy();br=Rotation.from_matrix(d.xmat[base].reshape(3,3))
        actual.append([np.r_[br.inv().apply(d.xpos[b]-bp),(br.inv()*Rotation.from_matrix(d.xmat[b].reshape(3,3))).as_quat()] for b in hands])
        base_positions.append(bp);tilts.append(np.rad2deg(np.arccos(np.clip(br.as_matrix()[2,2],-1,1))))
    actual=np.array(actual);base_positions=np.array(base_positions);tilts=np.array(tilts)
    stamps=np.array([r['wall_time'] for r in live]);indices=np.clip(np.searchsorted(recorded[:,0],stamps),0,len(recorded)-1)
    measured=actual[indices];desired=target['poses_xyzw'];limited=np.array([r['accepted_target_xyzw'] for r in live])
    yaw=target['inferred_hmd_yaw_rad'] if 'inferred_hmd_yaw_rad' in target else np.zeros(len(live))
    waist_goal=target['waist_goal_rad'] if 'waist_goal_rad' in target else np.full(len(live),meta['initial_q'][0])
    waist_measured=recorded[indices,2+int(m.joint('waist_yaw').qposadr[0])]
    raw_position,raw_angle=pose_error(measured,desired);limited_position,limited_angle=pose_error(measured,limited)
    success=np.array([r['success'] for r in audit]);position=np.array([r['position_error_mm'] for r in audit]);angle=np.array([r['orientation_error_deg'] for r in audit])
    commands=np.load(out/'commands.npy');a,b=coupled_halfspaces(list(URDF_JOINT_NAMES));coupled=commands[:,1:]@a.T-b
    limits=np.array([validate_urdf().joint_limits[n] for n in URDF_JOINT_NAMES]);q=commands[:,1:]
    axis=np.maximum(limits[:,0]-q,q-limits[:,1])[:,np.r_[:3,15:29]]
    summary=dict(source_frames=len(live),source_duration_s=meta['source_duration_s'],replay_duration_s=meta['duration_s'],
                 static_ik=dict(success_count=int(success.sum()),success_fraction=float(success.mean()),
                   collision_free_count=sum(r['collision_free'] for r in audit),feasible_count=sum(r['feasible'] for r in audit),
                   position_error_mm_all=stats(position),orientation_error_deg_all=stats(angle),
                   position_error_mm_success=stats(position[success]),orientation_error_deg_success=stats(angle[success]),
                   solve_ms=stats([r['solve_ms'] for r in audit]),failure_reasons=dict(Counter(x for r in audit if not r['success'] for x in r['reasons']))),
                 native=dict(accepted_waypoints=sum(r['success'] for r in live),
                   raw_target_reached_count=sum(r['success'] and r['fraction']==1 for r in live),
                   following_frames=sum(r['controller_mode']=='FOLLOWING' for r in live),
                   holds=dict(Counter(x for r in live if not r['success'] for x in r['reasons'])),
                   trips=dict(Counter(r['trip'] for r in live if r['trip'])),
                   raw_target_position_error_mm=stats(raw_position),raw_target_orientation_error_deg=stats(raw_angle),
                   accepted_target_position_error_mm=stats(limited_position),accepted_target_orientation_error_deg=stats(limited_angle),
                   command_count=len(commands),axis_limit_violation_count=int(np.any(axis>1e-6,axis=1).sum()),
                   coupled_limit_violation_count=int(np.any(coupled>1e-6,axis=1).sum()),
                   physics_realtime_ratio=float((recorded[-1,1]-recorded[0,1])/(t[-1]-t[0])),
                   physical_self_contact_frames=int(np.sum(recorded[:,-4]>0)),base_tilt_max_deg=float(tilts.max()),
                   base_xy_displacement_max_m=float(np.linalg.norm(base_positions[:,:2]-base_positions[0,:2],axis=1).max()),
                   fallen=bool(np.any(base_positions[:,2]<.5) or np.any(tilts>30))))
    summary['yaw']=dict(inferred=meta.get('inferred_hmd',False),
                        virtual_hmd_range_deg=np.rad2deg([yaw.min(),yaw.max()]).tolist(),
                        waist_goal_range_deg=np.rad2deg([waist_goal.min(),waist_goal.max()]).tolist(),
                        measured_waist_range_deg=np.rad2deg([waist_measured.min(),waist_measured.max()]).tolist(),
                        measured_waist_goal_error_deg=stats(np.rad2deg(abs(waist_measured-waist_goal))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'evaluation.npz',source_t=target['source_t'],raw_target_xyzw=desired,
                        measured_xyzw=measured,static_position_error_mm=position,static_orientation_error_deg=angle,
                        static_success=success,native_raw_position_error_mm=raw_position,native_raw_orientation_error_deg=raw_angle,
                        native_accepted_position_error_mm=limited_position,physics_wall_t=t,qpos=recorded[:,2:2+nq])
    header=['frame_index','source_time_s','replay_time_s','static_success','static_feasible','static_collision_free',
            'static_left_mm','static_right_mm','static_left_deg','static_right_deg','static_reasons',
            'native_waypoint_accepted','raw_target_fraction','measured_left_raw_mm','measured_right_raw_mm',
            'measured_left_raw_deg','measured_right_raw_deg','measured_left_accepted_mm','measured_right_accepted_mm','native_reasons','controller_mode',
            'virtual_hmd_yaw_deg','waist_goal_sdk_deg','measured_waist_sdk_deg']
    with (out/'per_target_errors.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header)
        for i,(r,s) in enumerate(zip(live,audit)):
            writer.writerow([r['frame_index'],r['source_t'],r['wall_time']-meta['start_wall'],s['success'],s['feasible'],s['collision_free'],
                 *position[i],*angle[i],';'.join(s['reasons']),r['success'],r['fraction'],*raw_position[i],*raw_angle[i],*limited_position[i],';'.join(r['reasons']),r['controller_mode'],
                 *np.rad2deg([yaw[i],waist_goal[i],waist_measured[i]])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    for i,side in enumerate(('Left','Right')):
        axes[0].plot(target['source_t'],position[:,i],label=side)
        axes[1].plot(target['source_t'],angle[:,i],label=side)
        axes[2].plot(target['source_t'],raw_position[:,i],label=side+' raw goal')
        axes[2].plot(target['source_t'],limited_position[:,i],linestyle='--',alpha=.7,label=side+' accepted waypoint')
    axes[0].axhline(8,color='red',linestyle=':');axes[0].set_ylabel('Static IK FK error (mm)');axes[0].set_yscale('symlog',linthresh=8)
    axes[0].set_ylim(bottom=0)
    axes[1].axhline(np.rad2deg(.08),color='red',linestyle=':');axes[1].set_ylabel('Static IK rotation (deg)')
    axes[2].set_ylabel('Native measured FK error (mm)')
    axes[3].plot(target['source_t'],np.rad2deg(waist_goal),label='Waist yaw goal (SDK)')
    axes[3].plot(target['source_t'],np.rad2deg(waist_measured),label='Waist yaw measured (SDK)')
    axes[3].set_ylabel('Yaw (deg)');axes[3].set_xlabel('Original dataset host time (s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(loc='upper right')
    fig.suptitle(meta.get('input_label','UMI relative replay')+' | X=identity | scale 1 | free-base WALK_LOW | no box')
    fig.tight_layout();fig.savefig(out/'errors.png',dpi=150);plt.close(fig)
    if not video:
        print(json.dumps(summary,indent=2),flush=True);return
    renderer=mujoco.Renderer(m,720,1280);camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,1.];camera.distance=2.8;camera.azimuth=135;camera.elevation=-12
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','24','-i','-',
                         '-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(out/'replay.mp4')],stdin=subprocess.PIPE)
    live_t=stamps-meta['start_wall']
    for sec in np.arange(0,t[-1],1/24):
        k=min(np.searchsorted(t,sec),len(t)-1);i=int(np.clip(np.searchsorted(live_t,sec,side='right')-1,0,len(live)-1))
        d.qpos[:]=recorded[k,2:2+nq];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
        br=Rotation.from_matrix(d.xmat[base].reshape(3,3));bp=d.xpos[base]
        if meta.get('inferred_hmd',False):
            point=bp+br.apply([0,0,.73]);end=point+br.apply([.28*np.cos(yaw[i]),.28*np.sin(yaw[i]),0])
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,1,0,1],dtype=np.float32))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.014,point,end);renderer.scene.ngeom+=1
        for side,color in enumerate(([0,1,1,1],[1,0,1,1])):
            point=bp+br.apply(desired[i,side,:3]);accepted_point=bp+br.apply(limited[i,side,:3])
            for p,c,radius in ((point,color,.018),(accepted_point,[1,.65,0,1],.009)):
                g=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),p,np.eye(3).ravel(),np.array(c,dtype=np.float32));renderer.scene.ngeom+=1
            orientation=br*Rotation.from_quat(desired[i,side,3:])
            for axis in range(3):
                g=renderer.scene.geoms[renderer.scene.ngeom];color_axis=np.r_[np.eye(3)[axis],1].astype(np.float32)
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CYLINDER,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color_axis)
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CYLINDER,.003,point,point+orientation.apply(np.eye(3)[axis]*.08));renderer.scene.ngeom+=1
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,128),fill=(20,25,30))
        label='ACCEPT' if live[i]['success'] else 'HOLD'
        text=[f'{meta.get("input_label","UMI")} | native vendor physics | source {live[i]["source_t"]:.2f}s / replay {sec:.2f}s | frame {live[i]["frame_index"]}',
              f'Static original-target IK: {"PASS" if success[i] else "FAIL"} | FK L/R {position[i,0]:.1f}/{position[i,1]:.1f} mm | {angle[i,0]:.1f}/{angle[i,1]:.1f} deg',
              f'Native {label} | measured vs RAW L/R {raw_position[i,0]:.1f}/{raw_position[i,1]:.1f} mm',
              f'Measured vs accepted waypoint L/R {limited_position[i,0]:.1f}/{limited_position[i,1]:.1f} mm | {live[i]["controller_mode"]}']
        for n,line in enumerate(text):draw.text((14,5+29*n),line,font=font,fill='white')
        if meta.get('inferred_hmd',False):
            draw.rectangle((0,128,1280,158),fill=(20,25,30))
            draw.text((14,129),f'Yellow = inferred HMD yaw {np.rad2deg(yaw[i]):+.1f} deg | waist SDK goal/measured {np.rad2deg(waist_goal[i]):+.1f}/{np.rad2deg(waist_measured[i]):+.1f} deg',font=font,fill='yellow')
        draw.rectangle((0,680,1280,720),fill=(20,25,30));draw.text((14,687),'Cyan/magenta = source EE goals + RGB axes | orange = accepted waypoint | X=identity, scale=1',font=font,fill='white')
        ff.stdin.write(im.tobytes())
        if sec==0:im.save(out/'first_pose.png')
    ff.stdin.close();assert ff.wait()==0;renderer.close()
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--no-video',action='store_true')
    args=parser.parse_args();process(args.out.resolve(),not args.no_video)
