"""Does the simulated D455 colour camera project like the calibrated real one?

    ~/isaacsim/python.sh scripts/check_camera.py

Floats a grid of small emissive markers in front of the wrist camera, renders it, and compares where each marker
lands with K + forward distortion from the calibration (stored on the camera prim by build_props.py).
"""
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np, omni.usd, omni.replicator.core as rep
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade

ROOT = Path(__file__).resolve().parents[1]
CAM = "/World/UR5e/wrist_3_link/d455_mount/D455/RSD455/Camera_OmniVision_OV9782_Color"
omni.usd.get_context().open_stage(str(ROOT / "ur5e_desk_scene.usda"))
for _ in range(40):
    app.update()
stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
cam = stage.GetPrimAtPath(CAM); W, H = cam.GetCustomDataByKey("r2s:resolution")
K = np.array(cam.GetCustomDataByKey("r2s:K")).reshape(3, 3); k1, k2, p1, p2, k3 = cam.GetCustomDataByKey("r2s:dist_opencv")
M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)).T @ np.diag([1.0, -1.0, -1.0, 1.0])     # optical (y down, z fwd) -> world

mat = UsdShade.Material.Define(stage, "/Markers/Glow"); sh = UsdShade.Shader.Define(stage, "/Markers/Glow/Shader")
sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset); sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl"); sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True); sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 1))
sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(20000.0); sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
Z = 0.30; expect = []
for i, x in enumerate(np.linspace(-0.70, 0.70, 7)):
    for j, y in enumerate(np.linspace(-0.50, 0.50, 5)):
        p = M @ np.array([x * Z, y * Z, Z, 1.0])
        s = UsdGeom.Sphere.Define(stage, f"/Markers/m{i}{j}"); s.CreateRadiusAttr(0.002); UsdGeom.XformCommonAPI(s).SetTranslate(Gf.Vec3d(*p[:3]))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)
        r2 = x * x + y * y; f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd, yd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        expect.append((K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2], K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]))
ann = rep.AnnotatorRegistry.get_annotator("HdrColor"); ann.attach([rep.create.render_product(CAM, (W, H))])
for _ in range(120):
    app.update()
rep.orchestrator.step()
img = np.asarray(ann.get_data())[..., :3].astype(np.float64)
glow = np.clip(img[..., 0] + img[..., 2] - 2 * img[..., 1], 0, None)             # magenta only
(ROOT / "renders").mkdir(exist_ok=True); Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(ROOT / "renders/check_camera.png")
vv, uu = np.mgrid[:H, :W]; err, err_nodist = [], []
for ue, ve, u0, v0 in expect:
    if not (12 < ue < W - 12 and 12 < ve < H - 12):
        continue
    win = (abs(uu - ue) < 10) & (abs(vv - ve) < 10); w = glow * win
    if w.sum() < 1e-6:
        print(f"[camera] marker expected at ({ue:.1f}, {ve:.1f}) not found", flush=True); continue
    um, vm = (w * (uu + 0.5)).sum() / w.sum(), (w * (vv + 0.5)).sum() / w.sum()
    err.append((um - ue, vm - ve)); err_nodist.append((um - u0, vm - v0))
err, err_nodist = np.array(err), np.array(err_nodist)
print(f"[camera] image mean {img.mean():.3f}; {len(err)} markers; rendered - calibrated model: mean {err.mean(0).round(2)} px, max {np.abs(err).max():.2f} px", flush=True)
print(f"[camera] rendered - same K without distortion: max {np.abs(err_nodist).max():.2f} px (how much the lens distortion moves points)", flush=True)
good = (np.abs(err).max(1) < 1.0).mean()                                       # the gripper fingers hide a couple of markers
print(f"[camera] markers within 1 px of the calibrated model: {good * 100:.0f} % (median error {np.median(np.abs(err)):.2f} px)", flush=True)
assert good > 0.85, "simulated camera does not match the calibration"
app.close()
