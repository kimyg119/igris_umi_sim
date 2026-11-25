"""Does the gripper work? One full close / open cycle of the Robotiq 2F-85 with nothing between the pads.

    ~/isaacsim/python.sh scripts/check_gripper.py

Checks that the finger joint reaches both ends of its range, that the pads meet when closed and open to the 85 mm
stroke, that both fingers move symmetrically and that the arm holds its pose meanwhile. Nothing is saved to the scene.
(Grasping a block is covered by replay_episode.py.)
"""
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np, omni.usd, omni.timeline, omni.physx
from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
G = "/World/UR5e/Gripper/Robotiq_2F_85/"
SPEED_DEG_S = 70.0            # real finger speed, as in replay_episode.py

omni.usd.get_context().open_stage(str(ROOT / "ur5e_desk_scene.usda"))
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
px = omni.physx.get_physx_interface()
fj = stage.GetPrimAtPath(G + "Joints/finger_joint"); target = fj.GetAttribute("drive:angular:physics:targetPosition")
closed_deg = float(fj.GetAttribute("physics:upperLimit").Get())
xc = UsdGeom.XformCache(); pads = {}
for link in ("left_inner_finger", "right_inner_finger"):       # fingertip pad meshes, in their link frame
    lw = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(G + link))
    for m in Usd.PrimRange(stage.GetPrimAtPath(G + link), Usd.TraverseInstanceProxies()):
        if m.IsA(UsdGeom.Mesh) and "fingertips" in str(m.GetPath()):
            pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get()); pads[link] = np.c_[pts, np.ones(len(pts))] @ np.array(xc.GetLocalToWorldTransform(m) * lw.GetInverse())


def world(path):
    t = px.get_rigidbody_transformation(path); x, y, z, w = t["rotation"]
    return np.array(Gf.Matrix4d().SetRotate(Gf.Quatd(w, x, y, z)).SetTranslateOnly(Gf.Vec3d(*t["position"])))


def measure():
    left, right = [(pads[k] @ world(G + k))[:, :3] for k in ("left_inner_finger", "right_inner_finger")]
    axis = right.mean(0) - left.mean(0); axis /= np.linalg.norm(axis); base = world(G + "base_link")[3, :3]
    gap = (right @ axis).min() - (left @ axis).max()                                  # between the pad faces
    skew = abs((left.mean(0) - base) @ axis + (right.mean(0) - base) @ axis)          # 0 if both fingers are equally far from the gripper axis
    dof = art.get_dof_positions().numpy()[0]
    return float(np.degrees(dof[fin])), gap * 1000, skew * 1000, float(np.degrees(np.abs(dof[arm] - q0)).max())


omni.timeline.get_timeline_interface().play()
for _ in range(30):
    app.update()
from isaacsim.core.experimental.prims import Articulation
art = Articulation("/World/UR5e"); names = list(art.dof_names); fin = names.index("finger_joint")
arm = [i for i, n in enumerate(names) if n.endswith("_joint") and n.split("_joint")[0] in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
q0 = art.get_dof_positions().numpy()[0][arm]
result = {"start (open)": measure()}; deg = 0.0
for name, goal in (("closed", closed_deg), ("open again", 0.0)):
    while abs(goal - deg) > 1e-6:
        deg += float(np.clip(goal - deg, -SPEED_DEG_S / 60, SPEED_DEG_S / 60)); target.Set(deg); app.update()
    for _ in range(60):
        app.update()
    result[name] = measure()
for name, (f, gap, skew, drift) in result.items():
    print(f"[gripper] {name:13s} finger joint {f:5.1f} deg  pad gap {gap:5.1f} mm  left/right asymmetry {skew:4.2f} mm  arm drift {drift:.2f} deg", flush=True)
f, gap, skew, drift = result["closed"]
assert abs(f - closed_deg) < 2 and gap < 2, "the gripper does not close"
assert all(abs(result[k][0]) < 1 and abs(result[k][1] - 85) < 3 for k in ("start (open)", "open again")), "the gripper does not open to its 85 mm stroke"
assert max(r[2] for r in result.values()) < 1 and max(r[3] for r in result.values()) < 0.5, "fingers asymmetric or the arm moved"
print("[gripper] OK", flush=True)
app.close()
