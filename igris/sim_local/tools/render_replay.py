#!/usr/bin/env python3
"""녹화된 로봇 상태(npz)를 직렬 시각화 모델로 재생해 영상(mp4)으로 렌더.

- 시각화 모델: xacro(parallel:=false) 직렬 모델 + RCV 핸드 그래프트
  (igris_c_v2_rcv_hand.xml 에서 asset/body 블록 이식)
- 재생: 관절공간 31개 + freejoint 베이스 포즈를 프레임마다 qpos에 설정, FK만.
- 렌더: EGL 오프스크린 → ffmpeg mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402


def graft_hands(serial_xml: str, hand_xml: str, out_xml: str) -> None:
    """RCV-hand MJCF에서 mesh asset과 손 body 블록을 직렬 모델에 이식."""
    st = ET.parse(serial_xml)
    sr = st.getroot()

    # meshdir 절대화 (직렬 모델은 상대경로)
    comp = sr.find("compiler")
    md = comp.get("meshdir", "")
    if md and not os.path.isabs(md):
        comp.set("meshdir", os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(serial_xml)),
                         "/robros/configs/description/mujoco", md)))

    if hand_xml == "none":
        # 손 없이: 프레임버퍼/메시디렉터리 준비만
        vis = sr.find("visual")
        if vis is None:
            vis = ET.SubElement(sr, "visual")
        g = vis.find("global")
        if g is None:
            g = ET.SubElement(vis, "global")
        g.set("offwidth", "1920"); g.set("offheight", "1080")
        st.write(out_xml)
        print(f"손 없는 viz 모델 -> {out_xml}")
        return

    hr = ET.parse(hand_xml).getroot()

    # 손 mesh asset 이식 (rcv_hand 메시는 절대경로 file 속성)
    s_asset = sr.find("asset")
    have = {m.get("name") for m in s_asset.findall("mesh")}
    n_mesh = 0
    for m in hr.find("asset").findall("mesh"):
        f = m.get("file", "")
        if "rcv_hand" in f and m.get("name") not in have:
            s_asset.append(m)
            n_mesh += 1

    # 손 body 블록 이식
    def find_body(root, name):
        for b in root.iter("body"):
            if b.get("name") == name:
                return b
        return None

    n_body = 0
    for side in ("l", "r"):
        hb = find_body(hr, f"{side}_hand_rcv")
        wc = find_body(sr, f"{side}_wrist_connector")
        assert hb is not None and wc is not None, (side, hb is None, wc is None)
        wc.append(hb)
        n_body += 1

    # 오프스크린 프레임버퍼 확장
    vis = sr.find("visual")
    if vis is None:
        vis = ET.SubElement(sr, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", "1920"); g.set("offheight", "1080")

    st.write(out_xml)
    print(f"그래프트 완료: mesh {n_mesh}개, hand body {n_body}개 -> {out_xml}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--serial", default="/igris/sim_local/viz_serial_base.xml")
    p.add_argument("--hands", default="/igris/sim_local/igris_c_v2_rcv_hand.xml")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    viz_xml = "/tmp/viz_with_hands.xml"
    graft_hands(args.serial, args.hands, viz_xml)

    model = mujoco.MjModel.from_xml_path(viz_xml)
    data = mujoco.MjData(model)
    print(f"viz 모델: nbody={model.nbody} ngeom={model.ngeom} nq={model.nq}")

    d = np.load(args.npz)
    t, base, q = d["t"], d["base"], d["q"]
    names = [str(x) for x in d["joint_names"]]

    # 관절 이름 → qpos 주소
    jadr = {}
    for jn in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            jadr[jn] = model.jnt_qposadr[jid]
    missing = [jn for jn in names if jn not in jadr]
    print(f"매핑된 관절 {len(jadr)}/{len(names)}", f"누락: {missing}" if missing else "")

    # freejoint 주소 (베이스)
    free_adr = None
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_adr = model.jnt_qposadr[jid]
            break
    assert free_adr is not None, "freejoint 없음 (fixed:=false 확인)"

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -12.0, 135.0

    fps = max(1.0, (len(t) - 1) / max(t[-1] - t[0], 1e-6))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE)

    for i in range(len(t)):
        data.qpos[:] = 0.0
        data.qpos[free_adr:free_adr + 3] = base[i, :3]
        data.qpos[free_adr + 3:free_adr + 7] = base[i, 3:7]  # wxyz
        for jn, adr in jadr.items():
            data.qpos[adr] = q[i, names.index(jn)]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [base[i, 0], base[i, 1], max(0.6, base[i, 2] * 0.85)]
        renderer.update_scene(data, camera=cam)
        ff.stdin.write(renderer.render().tobytes())

    ff.stdin.close()
    ff.wait()
    print(f"영상 저장: {args.out}  ({len(t)}프레임 @ {fps:.1f}fps)")


if __name__ == "__main__":
    main()
