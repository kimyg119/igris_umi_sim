#!/usr/bin/env python3
"""리타게팅된 AMASS pkl을 GMR 텔레옵 페이로드로 발행한다.

retargeting/retargeted/**.pkl (retarget_shard.py 산출물)을
/gmr/teleop/retarget_frame 에 라이브 GMR과 동일한 스키마로 발행.
gmt_public 입장에서는 진짜 텔레옵과 구분되지 않는다.

pkl 스키마 (tools/render_motion.py 주석으로 확인):
  fps, root_pos(N,3), root_rot(N,4 **xyzw**), dof_pos(N,29 — 목 없음),
  local_body_pos(N,34,3 — 루트 로컬), link_body_list(34)

변환:
  dof 29 → 31 : GMR 관절 순서가 시스템 순서의 앞 29와 동일(허리3+왼다리6+
                오른다리6+왼팔7+오른팔7), 목 2개는 0 패딩
  root_rot    : xyzw → wxyz
  속도        : 유한차분. 각속도는 월드 프레임 (수신기가 월드로 해석)
  local_body_pos → keybody_pos_local (이미 루트 로컬임을 확인)

텔레옵 녹화 청크 (record_meta.source == xrobot_teleop_to_robot_w_hand,
xrobot_teleop_to_robot_w_hand.py --save_pkl_*):
  root_rot 이미 wxyz(qpos[3:7]), dof_pos 31(목 포함), keybody_pos_local(N,33,3),
  local_body_pos=None. base_link 이 없지만 수신기가 [0,0,0]으로 삽입한다.
  fps 키는 --save_pkl_fps 라벨(기본 30)일 뿐 프레임은 매 스텝 저장되므로 실제
  프레임률은 record_start/end_unix 로 복원한다(저장된 속도는 dt=1/30 기준이라
  버리고 실제 dt 로 재계산).

--start_file 을 주면 그 파일이 생길 때까지 첫 프레임(속도 0)을 홀드하고, 생기면
처음부터 재생한다(기동 시퀀스 뒤 원하는 순간에 시작시키기 위함). --no-loop 면
끝에서 마지막 프레임을 유지한다.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj_wxyz(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def world_angvel(q_wxyz, dt):
    """월드 프레임 각속도: q_{t+1} = Δq_w ⊗ q_t → Δq_w의 축각/dt."""
    n = q_wxyz.shape[0]
    w = np.zeros((n, 3))
    if n < 2:
        return w
    dq = quat_mul_wxyz(q_wxyz[1:], quat_conj_wxyz(q_wxyz[:-1]))
    # 부호 정규화 (w>=0)
    sign = np.where(dq[:, :1] < 0, -1.0, 1.0)
    dq = dq * sign
    angle = 2.0 * np.arccos(np.clip(dq[:, 0], -1.0, 1.0))
    axis_norm = np.linalg.norm(dq[:, 1:], axis=1, keepdims=True)
    axis = np.where(axis_norm > 1e-12, dq[:, 1:] / np.maximum(axis_norm, 1e-12), 0.0)
    w[1:] = axis * angle[:, None] / dt
    w[0] = w[1] if n > 1 else 0.0
    return w


class PklPublisher(Node):
    def __init__(self, args):
        super().__init__("gmr_pkl_pub")
        with open(args.pkl, "rb") as f:
            d = pickle.load(f)

        meta = d.get("record_meta") or {}
        root_pos = np.asarray(d["root_pos"], dtype=np.float64)
        n = root_pos.shape[0]
        if meta.get("source") == "xrobot_teleop_to_robot_w_hand":
            schema = "teleop"
            root_rot = np.asarray(d["root_rot"], dtype=np.float64)          # 이미 wxyz
            dof31 = np.asarray(d["dof_pos"], dtype=np.float64)              # 31 (목 포함)
            kb_local = np.asarray(d["keybody_pos_local"], dtype=np.float64)
            span = float(meta.get("record_end_unix", 0)) - float(meta.get("record_start_unix", 0))
            fps = (n - 1) / span if (n > 1 and span > 0) else float(meta.get("target_fps", d["fps"]))
        else:
            schema = "amass"
            fps = float(d["fps"])
            root_rot = np.asarray(d["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]  # xyzw→wxyz
            dof29 = np.asarray(d["dof_pos"], dtype=np.float64)
            dof31 = np.concatenate([dof29, np.zeros((n, 2))], axis=1)  # 목 0 패딩
            kb_local = np.asarray(d["local_body_pos"], dtype=np.float64)

        dt = 1.0 / fps
        dof_vel = np.zeros_like(dof31)
        dof_vel[1:] = (dof31[1:] - dof31[:-1]) / dt
        root_vel = np.zeros_like(root_pos)
        root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
        root_angvel = world_angvel(root_rot, dt)

        body_names = list(d["link_body_list"])

        self.frames = []
        for i in range(n):
            self.frames.append({
                "fps": fps,
                "dof_pos": [dof31[i].tolist()],
                "dof_vel": [dof_vel[i].tolist()],
                "link_body_list": body_names,
                "keybody_pos_local": [kb_local[i].reshape(-1).tolist()],
                "root_pos": [root_pos[i].tolist()],
                "root_rot": [root_rot[i].tolist()],
                "root_height": float(root_pos[i][2]),
                "root_vel": [root_vel[i].tolist()],
                "root_angvel": [root_angvel[i].tolist()],
                "record_meta": {"unix_ms": 0, "default": False,
                                "source": "gmr_pkl_pub", "frame_idx": i},
            })

        # 시작 전 홀드용: 첫 프레임 자세, 속도 0
        self.hold = dict(self.frames[0])
        self.hold["dof_vel"] = [[0.0] * dof31.shape[1]]
        self.hold["root_vel"] = [[0.0, 0.0, 0.0]]
        self.hold["root_angvel"] = [[0.0, 0.0, 0.0]]
        self.hold["record_meta"] = dict(self.hold["record_meta"], hold=True)

        self.args = args
        self.step = 0
        self.started = not args.start_file
        self.pub = self.create_publisher(String, args.topic, qos_profile_sensor_data)
        self.get_logger().info(
            f"클립={args.pkl} [{schema}]  프레임={n} @ {fps:.2f} fps ({n/fps:.1f}s)  "
            f"-> {args.topic}  loop={args.loop}"
            + (f"  start_file={args.start_file} (첫 프레임 홀드 중)" if args.start_file else ""))
        self.create_timer(dt, self.tick)

    def tick(self):
        if not self.started:
            if os.path.exists(self.args.start_file):
                self.started = True
                self.step = 0
                self.get_logger().info("트리거 감지 — 재생 시작")
            else:
                self.hold["record_meta"]["unix_ms"] = int(time.time() * 1000)
                self.pub.publish(String(data=repr(self.hold)))
                return
        payload = self.frames[self.step]
        payload["record_meta"]["unix_ms"] = int(time.time() * 1000)
        self.pub.publish(String(data=repr(payload)))
        n = len(self.frames)
        self.step += 1
        if self.step >= n:
            self.step = 0 if self.args.loop else n - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--topic", default="/gmr/teleop/retarget_frame")
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   help="끝에서 마지막 프레임 유지 (루프 랩 점프 방지)")
    p.add_argument("--start_file", default=None,
                   help="이 파일이 생길 때까지 첫 프레임 홀드, 생기면 처음부터 재생")
    args = p.parse_args()
    rclpy.init()
    node = PklPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
