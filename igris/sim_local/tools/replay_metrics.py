#!/usr/bin/env python3
"""sim 재생 녹화(state_recorder.py npz)의 yaw 추종·발 들림 지표. 정제 전/후 A/B 용.

발 높이는 골반 기반 viz 모델(viz_serial_base.xml, 시뮬 베이스와 같은 프레임)에서 geom/mesh 를
떼어낸 FK 전용 모델로 잰다 — GMR MJCF 는 루트가 몸통(base_link)이라 시뮬 골반 포즈를 넣으면
허리 체인이 다리로 잘못 전파돼 가짜 들림이 나온다. 절대 높이가 아니라 발별 최솟값 대비 들림만 본다.
    python tools/replay_metrics.py --npz videos/box_pickup_clean.npz --ref_pkl ../retargeting/teleop/box_pickup.clean.pkl --t0 2.6
"""
import argparse
import os
import pickle
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "retargeting" / "tools"))
import motion_common as mc  # noqa: E402


def yaw_span(q_wxyz):
    y = np.unwrap(mc.yaw_from_quat(mc.quat_continuous(q_wxyz)))
    return float(np.degrees(y[-1] - y[0])), float(np.degrees(y.max() - y.min()))


def load_fk_model(viz_xml):
    """viz_serial_base.xml 에서 asset/geom 을 전부 제거해 메시 없이 컴파일되는 FK 전용 모델."""
    tree = ET.parse(viz_xml); root = tree.getroot()
    # 메시(asset/geom)와 FK 에 불필요한 요소 제거 — 액추에이터의 kv+dampratio 조합은 mujoco 3.2.7 전용이라
    # 최신 mujoco 에서 컴파일이 거부된다.
    for tag in ("asset", "actuator", "contact", "equality", "sensor", "tendon", "keyframe"):
        for el in root.findall(tag):
            root.remove(el)
    for parent in root.iter():
        for g in list(parent.findall("geom")):
            parent.remove(g)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(ET.tostring(root, encoding="unicode"))
    try:
        model = mujoco.MjModel.from_xml_path(fh.name)
    finally:
        os.remove(fh.name)
    return model, mujoco.MjData(model)


def foot_heights_viz(model, data, base, q, joint_names):
    jadr = {jn: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joint_names}
    free = [model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("l_ankle_frame", "r_ankle_frame")]
    h = np.empty((len(base), 2))
    for i in range(len(base)):
        data.qpos[:] = 0
        data.qpos[free:free + 3] = base[i, :3]; data.qpos[free + 3:free + 7] = base[i, 3:7]
        for jn, a in jadr.items():
            data.qpos[a] = q[i, joint_names.index(jn)]
        mujoco.mj_forward(model, data)
        h[i] = data.xpos[feet, 2]
    return h


def lift_events(h, thresh=0.02):
    rel = h - h.min(axis=0)
    ev, mx = [], []
    for k in range(2):
        up = rel[:, k] > thresh
        n = int(np.sum(up[1:] & ~up[:-1]))
        ev.append(n); mx.append(float(rel[:, k].max()))
    return ev, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ref_pkl", required=True)
    ap.add_argument("--t0", type=float, required=True, help="녹화 기준 재생 시작 시각(s)")
    ap.add_argument("--viz_xml", default=str(Path(__file__).resolve().parents[1] / "viz_serial_base.xml"))
    args = ap.parse_args()
    d = np.load(args.npz)
    t, base, q = d["t"], d["base"], d["q"]
    joint_names = [str(x) for x in d["joint_names"]]
    with open(args.ref_pkl, "rb") as f:
        ref = pickle.load(f)
    n_ref = len(ref["root_pos"]); dur = (n_ref - 1) / float(ref["fps"])
    m = (t >= args.t0) & (t <= args.t0 + dur)
    model, data = load_fk_model(args.viz_xml)
    h = foot_heights_viz(model, data, base[m], q[m], joint_names)
    ev, mx = lift_events(h)
    sim_span, sim_range = yaw_span(base[m, 3:7])
    ref_span, ref_range = yaw_span(np.asarray(ref["root_rot"])[:, [3, 0, 1, 2]])
    print(f"재생 구간 {t[m][0]:.1f}~{t[m][-1]:.1f}s ({m.sum()} 샘플)")
    print(f"yaw 변화  sim {sim_span:+.0f}° (범위 {sim_range:.0f}°)  |  ref {ref_span:+.0f}° (범위 {ref_range:.0f}°)  → 추종률 {sim_span / ref_span * 100 if abs(ref_span) > 1 else float('nan'):.0f}%")
    print(f"발 들림(>2cm) 이벤트 L/R = {ev[0]}/{ev[1]}, 최대 들림 L/R = {mx[0] * 100:.1f}/{mx[1] * 100:.1f} cm")
    print(f"base z 최소 {base[m, 2].min():.3f} m (낙상 판정 < 0.5)")


if __name__ == "__main__":
    main()
