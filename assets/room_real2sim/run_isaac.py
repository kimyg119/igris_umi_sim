"""Open the USD interactively, or verify / record real PhysX motion headlessly.

Usage: $ISAAC_SIM_DIR/python.sh run_isaac.py [--gui] [--out PATH]
"""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui",action="store_true")
parser.add_argument("--physics-only",action="store_true")
parser.add_argument("--arrangement-only",action="store_true",help="Check standing/layout and render without pushing the loads")
parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"previews")
parser.add_argument("--scene",type=Path,default=Path(__file__).resolve().parent/"room_interactive.usd")
args,kit_args=parser.parse_known_args()

from isaacsim import SimulationApp
app=SimulationApp({"headless":not args.gui,"width":1600,"height":1200,
                   "renderer":"RayTracedLighting","anti_aliasing":3,
                   "extra_args":["--/exts/isaacsim.physics.newton/auto_switch_on_startup=false",
                                 "--/exts/isaacsim.core.simulation_manager/default_engine=physx"]})
import carb
import numpy as np
import omni.usd
import omni.replicator.core as rep
from PIL import Image
from isaacsim.core.api import SimulationContext
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics

settings=carb.settings.get_settings()
settings.set("/rtx/post/tonemap/op",4)
settings.set("/rtx/post/tonemap/filmIso",320.)
settings.set("/rtx/post/tonemap/whitepoint",6500.)
settings.set("/rtx/rendermode","RayTracedLighting")
settings.set("/physics/updateToUsd",True)
settings.set("/physics/updateVelocitiesToUsd",True)
SimulationManager.switch_physics_engine("physx")
omni.usd.get_context().open_stage(str(args.scene.resolve()))
for _ in range(12):
    app.update()
stage=omni.usd.get_context().get_stage()
args.arrangement_only=args.arrangement_only or bool(stage.GetPrimAtPath("/World/IGRIS_C"))
settings.set("/physics/mouseInteractionEnabled",True)
settings.set("/physics/mouseGrab",True)
settings.set("/physics/mouseGrabIgnoreInvisible",False)
args.out.mkdir(parents=True,exist_ok=True)

if args.gui:
    from omni.kit.viewport.utility import get_active_viewport
    vp=get_active_viewport()
    if vp:
        vp.set_active_camera("/World/Cameras/Overview")
    print("GUI_READY: press Play, then Shift + left-drag to grab a box. Scene="+str(args.scene),flush=True)
    while app.is_running():
        app.update()
    app.close()
    raise SystemExit

report={"engine":"physx","physics_hz":240,"settle_seconds":5.0,"renders":{}}
try:
    layout=json.loads((args.scene.parent/"layout.json").read_text())
    paths=["/World/Boxes/"+b["name"] for b in layout["boxes"]]
    boxes=RigidPrim(paths)
    cart=None if args.arrangement_only else RigidPrim("/World/Rolltainer/Chassis")
    cache=UsdGeom.XformCache()
    authored=np.array([cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation() for p in paths])
    robot=RigidPrim("/World/IGRIS_C/Model/pelvis") if stage.GetPrimAtPath("/World/IGRIS_C") else None
    robot_authored=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/IGRIS_C/Model/pelvis")).ExtractTranslation()) if robot else None
    sim=SimulationContext(physics_dt=1/240,rendering_dt=1/60,stage_units_in_meters=1,
                          physics_prim_path="/World/PhysicsScene",device="cpu")
    sim.initialize_physics()
    sim.play()
    for _ in range(8):
        sim.step(render=False)
    start=boxes.get_world_poses()[0].numpy().copy()
    for i in range(1200):
        sim.step(render=False)
        if i%240==0:
            print(f"Settling {i//240+1}/5",flush=True)
    settled,orient=boxes.get_world_poses()
    settled=settled.numpy().copy()
    orient=orient.numpy().copy()
    drift=np.linalg.norm(settled-start,axis=1)
    report["max_settle_displacement_m"]=float(drift.max())
    report["max_displacement_from_authored_m"]=float(np.linalg.norm(settled-authored,axis=1).max())
    report["box_positions_after_settle"]={b["name"]:p.tolist() for b,p in zip(layout["boxes"],settled)}
    report["stable_stack"]=bool(drift.max()<.025 and settled[:,2].min()>.08)
    if robot:
        xyz,quat=robot.get_world_poses()
        xyz,quat=xyz.numpy()[0],quat.numpy()[0]
        report["robot"]={"pelvis_position_m":xyz.tolist(),"pelvis_quaternion_wxyz":quat.tolist(),
            "displacement_from_authored_m":float(np.linalg.norm(xyz-robot_authored)),
            "standing":bool(xyz[2]>.75 and np.linalg.norm(xyz[:2])<.05 and abs(quat[0])>.98),
            "free_base":True,"controller":"neutral joint position drives only"}
    report["pass"]=bool(report["stable_stack"] and (not robot or report["robot"]["standing"]))
    print("SETTLE",json.dumps({k:v for k,v in report.items() if k not in ("box_positions_after_settle","renders")}),flush=True)
    (args.out/"simulation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if args.physics_only:
        (args.out/"simulation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
        app.close(exit_code=0 if report["pass"] else 1)
        raise SystemExit(0 if report["pass"] else 1)
    physics_pass=report["pass"]
    report["pass"]=False
    assert report["stable_stack"],"Stack stability failed before rendering"
    sim.pause()
    # Capture the original arrangement after settling; no poses are baked into USD.
    cameras=("Overview","RobotFront","RobotBack","Overhead") if args.arrangement_only else ("Overview","Pallet","Rolltainer","Overhead")
    for name in cameras:
        product=rep.create.render_product("/World/Cameras/"+name,(1600,1200))
        rgb=rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        for _ in range(40):
            app.update()
        rep.orchestrator.step(rt_subframes=32,pause_timeline=True)
        data=np.asarray(rgb.get_data())[:,:,:3]
        imagepath=args.out/(name.lower()+".png")
        Image.fromarray(data).save(imagepath)
        report["renders"][name]={"mean":float(data.mean()),"std":float(data.std()),"bytes":imagepath.stat().st_size}
        assert 30<data.mean()<220 and data.std()>15,report["renders"][name]
        print("RENDER",name,report["renders"][name],flush=True)
        rgb.detach([product])
        product.destroy()
    if args.arrangement_only:
        report["pass"]=physics_pass
        assert report["pass"],"Robot standing validation failed; inspect simulation_validation.json"
        print("ARRANGEMENT_RESULT",json.dumps({k:v for k,v in report.items() if k!="box_positions_after_settle"}),flush=True)
        raise SystemExit(0)
    # PhysX-driven box perturbation: initial velocity only, never authored animation.
    product=rep.create.render_product("/World/Cameras/Overview",(1280,960))
    rgb=rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for _ in range(30):
        app.update()
    sim.play()
    cap=paths.index("/World/Boxes/Pallet_Cap")
    small=paths.index("/World/Boxes/Small_RightFront")
    boxes.set_velocities(linear_velocities=[[0,-1.5,.5],[0,-1.3,.8]],indices=[cap,small])
    cart_before=cart.get_world_poses()[0].numpy().copy()
    frames=args.out/"frames"
    frames.mkdir(exist_ok=True)
    for frame in range(90):
        for _ in range(8):
            # A modest, continuous force on the chassis exercises wheel / swivel joints.
            if 30<=frame<75:
                cart.apply_forces([[0,-18,0]])
            sim.step(render=False)
        sim.render()
        data=np.asarray(rgb.get_data())
        if data.size:
            Image.fromarray(data[:,:,:3]).save(frames/f"frame_{frame:04d}.png")
        if frame%30==0:
            print("INTERACTION_FRAME",frame,flush=True)
    after=boxes.get_world_poses()[0].numpy()
    cart_after=cart.get_world_poses()[0].numpy()
    report["box_interaction"]={"large_displacement_m":float(np.linalg.norm(after[cap]-settled[cap])),
        "small_displacement_m":float(np.linalg.norm(after[small]-settled[small])),
        "method":"initial velocity followed by gravity and contact, no pose animation"}
    report["cart_push"]={"force_N":[0,-18,0],"duration_s":1.5,
        "displacement_m":float(np.linalg.norm(cart_after-cart_before)),
        "before":cart_before.tolist(),"after":cart_after.tolist()}
    report["pass"]=bool(report["stable_stack"] and report["box_interaction"]["large_displacement_m"]>.10
        and report["box_interaction"]["small_displacement_m"]>.10 and report["cart_push"]["displacement_m"]>.03)
    print("RESULT",json.dumps({k:v for k,v in report.items() if k!="box_positions_after_settle"}),flush=True)
    (args.out/"simulation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    assert report["pass"],"Physics validation failed; inspect simulation_validation.json"
finally:
    (args.out/"simulation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    app.close(exit_code=0 if report.get("pass",False) else 1)
