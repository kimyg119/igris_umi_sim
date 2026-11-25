"""Replay a real episode in ur5e_desk_scene.usda and check the grasp and the wrist-camera view.

    ~/isaacsim/python.sh scripts/replay_episode.py [episode_index]      # after scripts/fetch_episode.py

The arm follows the recorded joint states (30 Hz, interpolated to the 60 Hz app step, 240 Hz physics); the gripper
follows the recorded open/close command at the real finger speed and stalls on the block by force, as the real one does.
Outputs in renders/replay_epXXX/: sim wrist frames (sim/ = scene-linear -> sRGB; sim_d455/ = with the real stream's
exposure/white-balance/tone fitted on the first frame, a camera-side effect that is not in the scene), real|sim video and
key-frame sheet, overview video, grasp close-ups, report.json.
Nothing is saved back to the scene.
"""
import json, subprocess, sys
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})

import numpy as np, omni.usd, omni.timeline, omni.physx, omni.replicator.core as rep
from PIL import Image, ImageFilter
from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
REF = ROOT / f"assets/reference/episode_{EP:03d}"; OUT = ROOT / f"renders/replay_ep{EP:03d}"; (OUT / "sim").mkdir(parents=True, exist_ok=True)
ep = json.load(open(REF / "episode.json")); state, action = np.array(ep["state"]), np.array(ep["action"]); steps = dict(ep["steps"])
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
FINGER_CLOSED_DEG, FINGER_SPEED_DEG_S, GRIP_LATENCY_S = 46.0, 70.0, 0.15      # fitted to the recorded gripper position ramp (0.01 -> 0.65 over frames 44..57)
CAM = "/World/UR5e/wrist_3_link/d455_mount/D455/RSD455/Camera_OmniVision_OV9782_Color"

omni.usd.get_context().open_stage(str(ROOT / "ur5e_desk_scene.usda"))
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
px = omni.physx.get_physx_interface(); tl = omni.timeline.get_timeline_interface()
drive = {n: stage.GetPrimAtPath(f"/World/UR5e/joints/{n}").GetAttribute("drive:angular:physics:targetPosition") for n in ARM}
finger = stage.GetPrimAtPath("/World/UR5e/Gripper/Robotiq_2F_85/Joints/finger_joint").GetAttribute("drive:angular:physics:targetPosition")
blocks = sorted(str(p.GetPath()) for p in stage.GetPrimAtPath("/World/Blocks").GetChildren())
GRIPPER = "/World/UR5e/Gripper/Robotiq_2F_85/"; pads = {}; _xc = UsdGeom.XformCache()
for link in ("left_inner_finger", "right_inner_finger"):   # fingertip pad meshes, in their link frame
    lw = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER + link))
    for m in Usd.PrimRange(stage.GetPrimAtPath(GRIPPER + link), Usd.TraverseInstanceProxies()):
        if m.IsA(UsdGeom.Mesh) and "fingertips" in str(m.GetPath()):
            pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get()); pads[link] = np.c_[pts, np.ones(len(pts))] @ np.array(_xc.GetLocalToWorldTransform(m) * lw.GetInverse())


def camera_products():
    over = UsdGeom.Camera.Define(stage, "/RenderCams/overview"); over.CreateFocalLengthAttr(22.0); over.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    eye, tgt = Gf.Vec3d(1.35, -1.75, 1.75), Gf.Vec3d(0.05, -0.15, 0.85); fwd = (tgt - eye).GetNormalized()
    right = Gf.Cross(fwd, Gf.Vec3d(0, 0, 1)).GetNormalized(); up = Gf.Cross(right, fwd)
    UsdGeom.Xformable(over).AddTransformOp().Set(Gf.Matrix4d(*right, 0, *up, 0, *(-fwd), 0, *eye, 1))
    out = {}
    close = UsdGeom.Camera.Define(stage, "/RenderCams/closeup"); close.CreateFocalLengthAttr(30.0); close.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    tgt = Gf.Vec3d(*px.get_rigidbody_transformation(blocks[0])["position"]) if False else Gf.Vec3d(*UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Pallet")).ExtractTranslation()) + Gf.Vec3d(-0.0775, 0.0775, 0.10)
    eye = tgt + Gf.Vec3d(-0.22, -0.30, 0.10); fwd = (tgt - eye).GetNormalized(); right = Gf.Cross(fwd, Gf.Vec3d(0, 0, 1)).GetNormalized(); up = Gf.Cross(right, fwd)
    UsdGeom.Xformable(close).AddTransformOp().Set(Gf.Matrix4d(*right, 0, *up, 0, *(-fwd), 0, *eye, 1))
    for name, path, res in (("wrist", CAM, (640, 480)), ("overview", "/RenderCams/overview", (1280, 800)), ("closeup", "/RenderCams/closeup", (800, 800))):
        a = rep.AnnotatorRegistry.get_annotator("HdrColor"); a.attach([rep.create.render_product(path, res)]); out[name] = a
    return out


def encode(ann):                                           # scene-linear -> sRGB, as a camera would store it
    lin = np.clip(np.asarray(ann.get_data())[..., :3].astype(np.float64), 0, 1)
    return (np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def pose(path):
    t = px.get_rigidbody_transformation(path); x, y, z, w = t["rotation"]
    m = np.eye(4); m[:3, :3] = np.array(Gf.Matrix3d(Gf.Quatd(w, x, y, z))).T; m[:3, 3] = t["position"]; return m


def fit_response(sim_lin, real_srgb):
    """Per-channel gain in linear RGB taking the sim render to the real camera's rendering of the same view, i.e. its
    auto-exposure and white balance. No offset and no channel mixing: blacks stay black, paint colours stay saturated."""
    pool = lambda a: a[:480, :640].reshape(24, 20, 32, 20, 3).mean((1, 3)).reshape(-1, 3)
    real = np.asarray(real_srgb, np.float64) / 255; real = np.where(real <= 0.04045, real / 12.92, ((real + 0.055) / 1.055) ** 2.4)
    X, Y = pool(np.clip(sim_lin, 0, 4)), pool(real); lit = X.min(1) > 0.05               # ignore near-black blocks
    return np.median(Y[lit] / X[lit], axis=0)


def d455_look(ann, gain):                                  # sim frame as the real stream would show it: response + slight softness
    out = np.clip(np.asarray(ann.get_data())[..., :3].astype(np.float64) * gain, 0, 1)
    img = Image.fromarray((np.where(out <= 0.0031308, 12.92 * out, 1.055 * out ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.7))


cams = camera_products()
tl.play()
for _ in range(90):                                        # let the stacks settle and the renderer converge at the home pose
    app.update()
rest = {b: pose(b) for b in blocks}
rep.orchestrator.step(pause_timeline=False)
RESP = fit_response(np.asarray(cams["wrist"].get_data())[..., :3], Image.open(REF / "frames/000.jpg"))
from isaacsim.core.experimental.prims import Articulation
art = Articulation("/World/UR5e"); names = list(art.dof_names); arm_idx = [names.index(n) for n in ARM]; fin_idx = names.index("finger_joint")

N = len(state); finger_deg = 0.0; log = []
for d in ("overview", "closeup", "sim_d455"):
    (OUT / d).mkdir(exist_ok=True)
for i in range(N):
    for sub in range(2):                                   # 30 Hz data -> 60 Hz app steps
        q = state[i, :6] if sub == 0 or i + 1 >= N else 0.5 * (state[i, :6] + state[i + 1, :6])
        for n, v in zip(ARM, np.degrees(q)):
            drive[n].Set(float(v))
        cmd = action[max(0, i - int(round(GRIP_LATENCY_S * 30))), 6]
        goal = FINGER_CLOSED_DEG if cmd > 0.5 else 0.0
        finger_deg += float(np.clip(goal - finger_deg, -FINGER_SPEED_DEG_S / 60, FINGER_SPEED_DEG_S / 60)); finger.Set(finger_deg)
        app.update()
    rep.orchestrator.step(pause_timeline=False)
    Image.fromarray(encode(cams["wrist"])).save(OUT / f"sim/{i:03d}.png"); d455_look(cams["wrist"], RESP).save(OUT / f"sim_d455/{i:03d}.png")
    Image.fromarray(encode(cams["overview"])).save(OUT / f"overview/{i:03d}.jpg", quality=90)
    if steps["grasp"] - 5 <= i <= steps["carry"] + 5:
        Image.fromarray(encode(cams["closeup"])).save(OUT / f"closeup/{i:03d}.jpg", quality=90)
    dof = art.get_dof_positions().numpy()[0]
    log.append(dict(q=dof[arm_idx], finger=float(np.degrees(dof[fin_idx])), tool=pose("/World/UR5e/wrist_3_link"), blocks={b: pose(b) for b in blocks},
                    pads={k: pose(GRIPPER + k) for k in pads}))
for _ in range(60):                                        # one more second so a dropped block comes to rest
    app.update()
final = {b: pose(b) for b in blocks}

# ---------------------------------------------------------------- report
disp = {b: np.linalg.norm(log[steps["carry"]]["blocks"][b][:3, 3] - rest[b][:3, 3]) for b in blocks}
picked = max(disp, key=disp.get)
rel = [np.linalg.inv(l["tool"]) @ l["blocks"][picked] for l in log]                 # block in the tool0 frame
hold = range(steps["carry"], steps["release"]); r0 = rel[steps["carry"]]             # carried: after the block has squared up in the pads
slip = max(np.linalg.norm(rel[i][:3, 3] - r0[:3, 3]) for i in hold) * 1000
tilt = max(np.degrees(np.arccos(np.clip((np.trace(r0[:3, :3].T @ rel[i][:3, :3]) - 1) / 2, -1, 1))) for i in hold)
ra = rel[steps["close"]]; squared = np.degrees(np.arccos(np.clip((np.trace(ra[:3, :3].T @ r0[:3, :3]) - 1) / 2, -1, 1)))
l = log[steps["carry"]]; Ti = np.linalg.inv(l["tool"])                               # tool x = closing direction
corners = np.array([[sx, sy, sz, 1] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * [0.0125, 0.0125, 0.0125, 1]
bx = (Ti @ l["blocks"][picked] @ corners.T)[0] * 1000
px_ = {k: (Ti @ l["pads"][k] @ pads[k].T)[0] * 1000 for k in pads}
face_pos = min(v.min() for v in px_.values() if v.mean() > 0); face_neg = max(v.max() for v in px_.values() if v.mean() < 0)
pad_pen = max(bx.max() - face_pos, face_neg - bx.min())                               # > 0: a pad is inside the block
basket = np.array(px.get_rigidbody_transformation("/World/Basket")["position"]); fp = final[picked][:3, 3] - basket
others = {b: np.linalg.norm(final[b][:3, 3] - rest[b][:3, 3]) * 1000 for b in blocks if b != picked}
report = dict(
    episode=EP, picked_block=picked.split("/")[-1], lifted_mm=round(disp[picked] * 1000, 1),
    arm_tracking_max_deg=round(float(np.degrees(np.abs(np.array([l["q"] for l in log]) - state[:, :6]).max())), 3),
    block_in_tool0_at_lift_mm=(r0[:3, 3] * 1000).round(1).tolist(),
    block_turned_while_closing_deg=round(float(squared), 1), block_shift_while_closing_mm=round(float(np.linalg.norm(r0[:3, 3] - ra[:3, 3]) * 1000), 1),
    slip_while_carrying_mm=round(float(slip), 2), tilt_while_carrying_deg=round(float(tilt), 2),
    pad_gap_holding_mm=round(float(face_pos - face_neg), 1), block_width_between_pads_mm=round(float(bx.max() - bx.min()), 1),
    pad_penetration_into_block_mm=round(float(pad_pen), 2),
    real_pad_gap_holding_mm=round(85 * (1 - float(state[steps["carry"], 6]) * 255 / 228), 1),   # Robotiq: 85 mm stroke, closed at 228/255
    finger_deg_holding=round(log[steps["carry"]]["finger"], 1), real_gripper_pos_holding=round(float(state[steps["carry"], 6]), 3),
    final_block_pos_in_basket_frame_mm=(fp * 1000).round(1).tolist(),
    landed_in_basket=bool(abs(fp[0]) < 0.105 and abs(fp[1]) < 0.150 and fp[2] < 0.06),
    other_blocks_max_move_mm=round(max(others.values()), 2), other_blocks_moved_over_10mm=sorted(b.split("/")[-1] for b, v in others.items() if v > 10))
report["grasp_stable"] = bool(report["lifted_mm"] > 50 and slip < 2 and tilt < 3 and pad_pen < 1.0 and report["landed_in_basket"] and not report["other_blocks_moved_over_10mm"])
small = lambda f: np.asarray(Image.open(f).convert("RGB").resize((80, 60), Image.BOX), np.float64)
mae = lambda d: float(np.mean([np.abs(small(REF / f"frames/{i:03d}.jpg") - small(OUT / f"{d}/{i:03d}.png")).mean() for i in range(N)]))
report["wrist_frame_mean_abs_diff_8bit"] = dict(raw_srgb=round(mae("sim"), 1), with_d455_response=round(mae("sim_d455"), 1))
report["d455_gain_linear_rgb"] = RESP.round(3).tolist()
yaw = [float(np.degrees(np.arctan2(r[1, 0], r[0, 0])) % 90) for r in rel]              # block yaw about the tool axis, mod its symmetry
report["series"] = dict(frame=list(range(N)), finger_deg=[round(l["finger"], 2) for l in log], real_gripper=state[:, 6].round(3).tolist(),
                        block_in_tool0_mm=[(r[:3, 3] * 1000).round(1).tolist() for r in rel], block_yaw_in_tool0_deg=np.round(yaw, 1).tolist())
json.dump(report, open(OUT / "report.json", "w"), indent=1)
for k, v in report.items():
    if k != "series":
        print(f"[replay] {k}: {v}", flush=True)

# ---------------------------------------------------------------- real | sim media
keys = [steps[k] + o for k, o in (("approach", 0), ("grasp", 0), ("close", 0), ("lift", 0), ("carry", 0), ("release", 2), ("home", -6))] + [N - 1]
sheet = Image.new("RGB", (640 * len(keys), 960))
for c, i in enumerate(keys):
    sheet.paste(Image.open(REF / f"frames/{i:03d}.jpg"), (c * 640, 0)); sheet.paste(Image.open(OUT / f"sim_d455/{i:03d}.png"), (c * 640, 480))
sheet.save(OUT / "real_top_sim_bottom.jpg", quality=88)
ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30"]
subprocess.run(ff + ["-i", str(REF / "frames/%03d.jpg"), "-framerate", "30", "-i", str(OUT / "sim_d455/%03d.png"), "-filter_complex", "hstack",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT / "real_vs_sim.mp4")])
subprocess.run(ff + ["-i", str(OUT / "overview/%03d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT / "overview.mp4")])
tl.stop(); app.close()
