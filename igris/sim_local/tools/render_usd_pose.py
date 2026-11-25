#!/usr/bin/env python3
"""IGRIS-C USD 를 Isaac Sim 헤드리스로 띄워 주어진 관절 자세로 놓고 PNG 로 렌더한다.

    ~/isaacsim/python.sh render_usd_pose.py --usd <robot.usd> --pose pose.json --out <dir>

pose.json: {"<usd joint name>": rad, ...}. 없는 관절은 0. 손·베이스 고정 관절은 건드리지 않는다.
물리는 중력 0 + 관절 드라이브 홀드로 "정지 화면" 만 만든다 — 접촉·낙상 검증이 아니다.
"""
import argparse, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--pose", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--settle", type=int, default=200)
a = ap.parse_args()
W, H = (int(v) for v in a.res.split("x"))

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": "RayTracedLighting"})

import numpy as np, carb, omni.usd, omni.timeline
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf, Sdf
from PIL import Image

pose = {k: float(v) for k, v in json.load(open(a.pose)).items()}
os.makedirs(a.out, exist_ok=True)

# ---- stage ----
ctx = omni.usd.get_context()
assert ctx.open_stage(os.path.abspath(a.usd)), "open_stage failed"
for _ in range(10):
    app.update()
stage = ctx.get_stage()
root = stage.GetDefaultPrim()
art_paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
assert len(art_paths) == 1, f"articulation roots: {art_paths}"
print("[pose] root", root.GetPath(), "articulation", art_paths[0])

# 중력 0 (자유 베이스가 떨어지지 않게) + 31 회전 관절 드라이브 홀드
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityMagnitudeAttr().Set(0.0)
rev = {}
for p in stage.Traverse():
    if p.IsA(UsdPhysics.RevoluteJoint) and p.IsActive():
        rev[p.GetName()] = p
missing = [k for k in pose if k not in rev]
assert not missing, f"pose keys not in USD revolute joints: {missing}"
limit_warn = []
for name, prim in rev.items():
    q = pose.get(name, 0.0)
    deg = math.degrees(q)
    rj = UsdPhysics.RevoluteJoint(prim)
    lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
    if lo is not None and hi is not None and not (lo - 1e-3 <= deg <= hi + 1e-3):
        limit_warn.append((name, deg, lo, hi))
    drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(1.0e5)
    drv.CreateDampingAttr(1.0e3)
    drv.CreateTargetPositionAttr(deg)
for w in limit_warn:
    print("[pose] WARN out of USD limit:", w)

# 발 링크의 기본자세 월드 z 로 발바닥 오프셋을 잰다 (루트 골반 = 원점, bbox min z = 발바닥)
bbox0 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(root).ComputeAlignedRange()
foot_prims = [str(p.GetPath()) for p in root.GetChildren() if p.GetName().endswith("foot_original")]
xc = UsdGeom.XformCache()
foot_z0 = [xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f)).ExtractTranslation()[2] for f in foot_prims]
sole_off = min(foot_z0) - bbox0.GetMin()[2]
print(f"[pose] default bbox z [{bbox0.GetMin()[2]:.3f}, {bbox0.GetMax()[2]:.3f}] foot link z {foot_z0} sole_off {sole_off:.3f}")

# ---- 조명 · 톤맵 ----
dome = UsdLux.DomeLight.Define(stage, "/RenderLights/Dome"); dome.CreateIntensityAttr(150.0)
sun = UsdLux.DistantLight.Define(stage, "/RenderLights/Sun"); sun.CreateIntensityAttr(1800.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 30))
fill = UsdLux.SphereLight.Define(stage, "/RenderLights/Fill"); fill.CreateIntensityAttr(6000.0); fill.CreateRadiusAttr(0.3)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-2.0, 2.5, 2.0))
s = carb.settings.get_settings()
s.set("/rtx/rendermode", "RayTracedLighting")
s.set("/rtx/post/tonemap/op", 4); s.set("/rtx/post/tonemap/filmIso", 110.0)
s.set("/rtx/post/tonemap/whitepoint", 6500.0); s.set("/rtx/post/tonemap/enabled", True)
s.set("/rtx/post/aa/op", 3)

# ---- 물리 시작 → 자세 텔레포트 ----
tl = omni.timeline.get_timeline_interface(); tl.play()
for _ in range(5):
    app.update()
from isaacsim.core.experimental.prims import Articulation, XformPrim
art = Articulation(art_paths[0])
names = list(art.dof_names)
q = np.array([[pose.get(n, 0.0) for n in names]], dtype=np.float32)
print("[pose] dofs", len(names), "set:", sum(n in pose for n in names))
art.set_dof_positions(q); art.set_dof_velocities(np.zeros_like(q)); art.set_dof_position_targets(q)
for _ in range(30):
    app.update()
qa = np.asarray(art.get_dof_positions().numpy()).ravel()
err = np.abs(qa - q.ravel()); print(f"[pose] hold err max {err.max():.4f} rad ({names[int(err.argmax())]})")

feet = XformPrim(foot_prims)
fz = np.asarray(feet.get_world_poses()[0].numpy())[:, 2]
ground_z = float(fz.min() - sole_off)
rootp = np.asarray(art.get_world_poses()[0].numpy()).ravel()
print(f"[pose] posed foot link z {fz} → ground z {ground_z:.3f}; root {rootp}")

# 시각용 바닥 (충돌 없음)
gnd = UsdGeom.Cube.Define(stage, "/RenderGround"); gnd.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(gnd.GetPrim()); gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, ground_z - 0.01)); gx.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.02))
gnd.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.43, 0.41)])

# ---- 카메라 ----
def look_at(eye, target, up=Gf.Vec3d(0, 0, 1)):
    eye, target = Gf.Vec3d(*eye), Gf.Vec3d(*target)
    fwd = (target - eye).GetNormalized()
    if abs(fwd * up) > 0.99: up = Gf.Vec3d(0, 1, 0)
    right = (fwd ^ up).GetNormalized(); cu = (right ^ fwd).GetNormalized()
    m = Gf.Matrix4d()
    m.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0)); m.SetRow(1, Gf.Vec4d(cu[0], cu[1], cu[2], 0))
    m.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0)); m.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    return m
cam = UsdGeom.Camera.Define(stage, "/RenderCam"); cam.CreateFocalLengthAttr(26.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
cx = UsdGeom.Xformable(cam.GetPrim()); cx.ClearXformOpOrder(); mop = cx.AddTransformOp()
center = (float(rootp[0]), float(rootp[1]), ground_z + 0.5 * (bbox0.GetMax()[2] - bbox0.GetMin()[2]))
views = {"front34": (2.7, -2.2, 0.45), "left": (0.0, 3.3, 0.35), "front": (3.3, 0.0, 0.35)}

rp = rep.create.render_product("/RenderCam", (W, H))
rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
tag = os.path.splitext(os.path.basename(a.usd))[0]
report = {"usd": os.path.abspath(a.usd), "articulation": art_paths[0], "dof_names": names,
          "pose_rad": {n: float(v) for n, v in zip(names, q.ravel())}, "hold_err_max_rad": float(err.max()),
          "ground_z": ground_z, "limit_warnings": limit_warn, "frames": {}}
first = True
for vname, off in views.items():
    eye = (center[0] + off[0], center[1] + off[1], center[2] + off[2])
    mop.Set(look_at(eye, center))
    for _ in range(a.settle if first else max(60, a.settle // 3)):
        app.update()
    first = False
    rep.orchestrator.step(); app.update()
    data = rgb.get_data()
    img = np.asarray(data)[:, :, :3]
    mean, mx = float(img.mean()), int(img.max())
    ok = mx > 0 and 10 < mean < 220
    fn = os.path.join(a.out, f"{tag}_{vname}.png"); Image.fromarray(img).save(fn)
    report["frames"][vname] = {"file": fn, "mean": mean, "max": mx, "ok": ok, "bytes": os.path.getsize(fn)}
    print(f"[render] {vname}: mean {mean:.1f} max {mx} {'OK' if ok else 'BAD'} → {fn}")
json.dump(report, open(os.path.join(a.out, f"{tag}_pose_report.json"), "w"), indent=1)
tl.stop(); app.close()
