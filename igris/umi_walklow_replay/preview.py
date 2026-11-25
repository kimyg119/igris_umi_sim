"""Render still images of a scene USD in Isaac Sim without stepping physics."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--camera', action='append', help='Camera prim path; default: every /World/Cameras child')
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=960)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height, 'renderer': 'RayTracedLighting',
                     'extra_args': ['--/exts/isaacsim.physics.newton/auto_switch_on_startup=false',
                                    '--/exts/isaacsim.core.simulation_manager/default_engine=physx']})
import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics

try:
    # Physics schemas are removed in a flattened copy so nothing can simulate.
    stage = Usd.Stage.Open(str(args.scene.resolve()))
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        else:
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
    cameras = args.camera or [str(p.GetPath()) for p in stage.GetPrimAtPath('/World/Cameras').GetChildren()
                              if p.IsA(UsdGeom.Camera)]
    flat = args.out / 'preview_scene.usdc'
    stage.Flatten().Export(str(flat))
    omni.usd.get_context().open_stage(str(flat))
    omni.timeline.get_timeline_interface().stop()
    for _ in range(20):
        app.update()
    settings = carb.settings.get_settings()
    settings.set('/rtx/post/tonemap/op', 4); settings.set('/rtx/post/tonemap/filmIso', 320.)
    sensors = []
    for path in cameras:
        product = rep.create.render_product(path, (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        sensors.append((path, rgb))
    for _ in range(60):
        app.update()
    rep.orchestrator.step(rt_subframes=16, delta_time=0., pause_timeline=True)
    for path, rgb in sensors:
        image = np.asarray(rgb.get_data())[:, :, :3]
        target = args.out / (path.rsplit('/', 1)[-1] + '.png')
        Image.fromarray(image).save(target)
        print('PREVIEW', target, float(image.mean()), float(image.std()), flush=True)
finally:
    app.close()
