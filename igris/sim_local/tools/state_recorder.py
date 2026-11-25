#!/usr/bin/env python3
"""시뮬 로봇 상태(/mujoco/joint_states + /mujoco/base_odom)를 npz로 녹화.

오프라인 영상 렌더용. 각 샘플: 시각, base pos(3)+quat wxyz(4), 관절 이름별 각도.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


class Recorder(Node):
    def __init__(self, args):
        super().__init__("state_recorder")
        self.args = args
        self.t0 = time.monotonic()
        self.joint_names = None
        self.last_joint = None
        self.last_odom = None
        self.rows = []          # (t, base7, q31)
        self.create_subscription(JointState, args.joint_topic,
                                 self.on_joint, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/mujoco/base_odom",
                                 self.on_odom, qos_profile_sensor_data)
        self.create_timer(1.0 / args.rate, self.sample)
        self.get_logger().info(f"녹화 시작: {args.duration}s @ {args.rate}Hz -> {args.out}")

    SYSTEM_JOINTS = [
        "waist_yaw","waist_roll","waist_pitch",
        "l_hip_pitch","l_hip_roll","l_hip_yaw","l_knee_pitch","l_ankle_pitch","l_ankle_roll",
        "r_hip_pitch","r_hip_roll","r_hip_yaw","r_knee_pitch","r_ankle_pitch","r_ankle_roll",
        "l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow_pitch",
        "l_wrist_yaw","l_wrist_roll","l_wrist_pitch",
        "r_shoulder_pitch","r_shoulder_roll","r_shoulder_yaw","r_elbow_pitch",
        "r_wrist_yaw","r_wrist_roll","r_wrist_pitch",
        "neck_yaw","neck_pitch",
    ]

    def on_joint(self, msg):
        if self.joint_names is None:
            self.joint_names = list(msg.name) if msg.name else list(self.SYSTEM_JOINTS[:len(msg.position)])
        self.last_joint = np.asarray(msg.position, dtype=np.float64)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation      # ROS는 xyzw
        self.last_odom = np.array([p.x, p.y, p.z, o.w, o.x, o.y, o.z])

    def sample(self):
        t = time.monotonic() - self.t0
        if self.last_joint is not None and self.last_odom is not None:
            self.rows.append((t, self.last_odom.copy(), self.last_joint.copy()))
        if t >= self.args.duration:
            self.finish()

    def finish(self):
        if not self.rows:
            self.get_logger().error("샘플 0개 — 토픽 미수신")
            raise SystemExit(1)
        ts = np.array([r[0] for r in self.rows])
        base = np.stack([r[1] for r in self.rows])
        qs = np.stack([r[2] for r in self.rows])
        np.savez(self.args.out, t=ts, base=base, q=qs,
                 joint_names=np.array(self.joint_names))
        self.get_logger().info(f"저장: {self.args.out}  샘플 {len(ts)}개, "
                               f"{ts[-1]:.1f}s, 관절 {qs.shape[1]}개")
        raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--joint_topic", default="/igris_c_001/rt/joint_state")
    args = p.parse_args()
    rclpy.init()
    node = Recorder(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
