"""Headless Isaac Sim check of ur5e_desk_scene.usda: renders + a short physics settle.

    ~/isaacsim/python.sh scripts/render_check.py [--physics]

Nothing is saved back to the scene; cameras live in the session layer. Outputs go to renders/.

In this Isaac Sim 6.0 build the Replicator "rgb" annotator is scene-linear (no tone curve, no gamma; only the
viewport applies ACES). So PNGs here are HdrColor with the sRGB transfer applied, i.e. what a camera would store.
"""
import json, sys
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})

import carb, numpy as np, omni.usd, omni.timeline, omni.replicator.core as rep
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "renders"; OUT.mkdir(exist_ok=True)
omni.usd.get_context().open_stage(str(ROOT / "ur5e_desk_scene.usda"))
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
_xc = UsdGeom.XformCache()                                  # authored poses, read before anything is simulated
authored = {str(p.GetPath()): (np.array(_xc.GetLocalToWorldTransform(p).ExtractTranslation()), _xc.GetLocalToWorldTransform(p).GetOrthonormalized().ExtractRotationQuat())
            for p in stage.Traverse() if p.HasAPI("PhysicsRigidBodyAPI")}
s = carb.settings.get_settings()
print("[check] tonemap op", s.get("/rtx/post/tonemap/op"), "filmIso", s.get("/rtx/post/tonemap/filmIso"),
      "rendermode", s.get("/rtx/rendermode"), flush=True)


def look_at(eye, target, up=Gf.Vec3d(0, 0, 1)):
    eye, target = Gf.Vec3d(*eye), Gf.Vec3d(*target)
    fwd = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up).GetNormalized(); cam_up = Gf.Cross(right, fwd)
    return Gf.Matrix4d(right[0], right[1], right[2], 0, cam_up[0], cam_up[1], cam_up[2], 0, -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)


def camera(name, matrix, res, focal_mm, h_aperture=20.955):
    cam = UsdGeom.Camera.Define(stage, f"/RenderCams/{name}")
    cam.CreateFocalLengthAttr(focal_mm); cam.CreateHorizontalApertureAttr(h_aperture)
    cam.CreateVerticalApertureAttr(h_aperture * res[1] / res[0]); cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    UsdGeom.Xformable(cam).AddTransformOp().Set(matrix)
    rp = rep.create.render_product(str(cam.GetPath()), res)
    ann = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ("HdrColor",)}
    for a in ann.values():
        a.attach([rp])
    return ann


ref = json.load(open(ROOT / "assets/reference/phone_view.json"))
D = np.diag([1.0, -1.0, -1.0]); R = D @ np.array(ref["R_cam_to_plane"]) @ D          # USD camera axes in world, as columns
phone_eye = (-0.45, 0.30 + ref["junction_y_plane_m"], 0.73 + ref["cam_height_m"])
phone_m = Gf.Matrix4d(*R[:, 0], 0, *R[:, 1], 0, *R[:, 2], 0, *phone_eye, 1)
views = {
    "overview": camera("overview", look_at((1.45, -1.95, 1.85), (0.05, -0.2, 0.82)), (1600, 1000), 22.0),
    "cell_front": camera("cell_front", look_at((0.05, -1.75, 1.45), (0.03, 0.0, 0.85)), (1600, 1000), 26.0),
    "pallet_closeup": camera("pallet_closeup", look_at((-0.42, -0.36, 0.98), (-0.13, -0.02, 0.79)), (1400, 1000), 30.0),
    "wrist_closeup": camera("wrist_closeup", look_at((-0.33, -0.40, 1.36), (0.03, -0.12, 1.19)), (1200, 1000), 35.0),
    "basket_closeup": camera("basket_closeup", look_at((0.62, -0.40, 1.08), (0.215, 0.0, 0.775)), (1200, 1600), 35.0),
    "phone_view": camera("phone_view", phone_m, (ref["width"], ref["height"]), 20.955 * ref["f_px"] / ref["width"]),
}
HIDE = ("/World/UR5e", "/World/Basket", "/World/Pallet", "/World/Blocks", "/World/RobotStand")


def settle(n=220):
    for _ in range(n):
        app.update()
    rep.orchestrator.step()


def encode(ann):
    lin = np.clip(np.asarray(ann["HdrColor"].get_data())[..., :3].astype(np.float64), 0, 1)
    return (np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def save(tag=""):
    for name, ann in views.items():
        rgb = encode(ann)
        Image.fromarray(rgb).save(OUT / f"{name}{tag}.png")
        print(f"[check] {name}{tag}: mean {rgb.mean():.0f} max {rgb.max()} p99 {np.percentile(rgb, 99):.0f}", flush=True)


def phone_ratio(tag):
    """Scene-linear desk / wall radiance in the phone view, same image regions as measured in the video frame."""
    hdr = np.asarray(views["phone_view"]["HdrColor"].get_data())[..., :3].astype(np.float64)
    desk, wall = hdr[900:1300, 300:800].mean((0, 1)), hdr[20:170, 20:220].mean((0, 1))
    real = np.array(ref["desk_lin"]) / np.array(ref["wall_lin"])
    print(f"[check] {tag} desk/wall radiance sim {np.round(desk / wall, 3)} vs video {np.round(real, 3)}; sim wall {np.round(wall, 1)} desk {np.round(desk, 1)}", flush=True)


settle(); save(); phone_ratio("with objects:")
# same phone view without robot/basket: the video was shot on the empty desk
for p in HIDE:
    if stage.GetPrimAtPath(p):
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
settle(120)
sim = Image.fromarray(encode(views["phone_view"])); sim.save(OUT / "phone_view_empty.png"); phone_ratio("empty desk:")
real = Image.open(ROOT / "assets/reference/phone_frame.png")                      # left: video frame, right: sim from the same pose
pair = Image.new("RGB", (real.width + sim.width, real.height)); pair.paste(real, (0, 0)); pair.paste(sim, (real.width, 0)); pair.save(OUT / "phone_real_vs_sim.jpg", quality=90)
for p in HIDE:
    if stage.GetPrimAtPath(p):
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeVisible()

if "--physics" in sys.argv:
    import omni.physx
    px = omni.physx.get_physx_interface()
    omni.timeline.get_timeline_interface().play()
    for _ in range(180):
        app.update()
    worst = 0.0
    for b, (t0, q0) in authored.items():                    # drift from the authored (pre-simulation) pose
        tr = px.get_rigidbody_transformation(b); x, y, z, w = tr["rotation"]
        ang = np.degrees(2 * np.arccos(min(1.0, abs(Gf.Dot(Gf.Quatd(w, x, y, z).GetNormalized().GetImaginary(), q0.GetImaginary()) + w * q0.GetReal()))))
        mm = np.linalg.norm(np.array(tr["position"]) - t0) * 1000; worst = max(worst, mm, ang)
        print(f"[check] physics 3 s: {b.split('/World/')[-1]:36s} moved {mm:6.2f} mm  rotated {ang:6.2f} deg  z {tr['position'][2]:.4f}", flush=True)
    print(f"[check] physics worst drift {worst:.2f} (mm or deg)", flush=True)
    settle(60); save("_after_physics")
    omni.timeline.get_timeline_interface().stop()
app.close()
