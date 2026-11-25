#!/usr/bin/env python3
"""녹화 모션 ONNX를 GMR 텔레옵 페이로드로 변환해 ROS2로 발행한다.

PICO 헤드셋 없이 텔레옵(ros2) 경로 전체를 검증하기 위한 테스트 하네스.
GMR의 redis_to_ros2_bridge.py 와 같은 토픽/포맷으로 발행하므로,
igris_c_gmt_public 입장에서는 진짜 GMR과 구분되지 않는다.

매핑은 src/motion/frame_source.cpp 의 onnx_replay 경로를 그대로 따른다:
  joint_pos/joint_vel        -> dof_pos / dof_vel
  body_pos_b                 -> keybody_pos_local
  body_names(metadata)       -> link_body_list
  body_pos_w[base_link]      -> root_pos
  body_quat_w[base_link]     -> root_rot (wxyz)
  body_lin_vel_b[base_link]  -> root_vel    (월드 프레임으로 회전시켜 발행)
  body_ang_vel_b[base_link]  -> root_angvel (동일)

수신측(ros_receiver.cpp)은 root_vel/root_angvel을 월드 프레임으로 보고
앵커/바디 프레임 사본을 역산한다. 그래서 여기서 바디->월드로 돌려 보낸다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z),     s * (x * z + w * y)],
        [s * (x * y + w * z),     1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y),     s * (y * z + w * x),     1 - s * (x * x + y * y)],
    ])


class MotionOnnx:
    def __init__(self, path: str):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.body_names = [s for s in meta["body_names"].split(",") if s]
        self.fps = float(meta.get("motion_fps", 50.0))
        self.num_frames = int(meta.get("motion_num_frames", 0))
        if "base_link" not in self.body_names:
            raise RuntimeError("모션 ONNX metadata의 body_names에 base_link가 없습니다")
        self.base = self.body_names.index("base_link")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # 모델이 요구하는 랭크에 맞춘다 (이 모델은 time_step 랭크 2).
        self.input_rank = len(inp.shape) if inp.shape else 1
        self.output_names = [o.name for o in self.session.get_outputs()]

    def frame(self, step: int) -> dict:
        outs = self.session.run(
            self.output_names,
            {self.input_name: np.full((1,) * self.input_rank, step, dtype=np.int64)},
        )
        o = {n: np.asarray(v).reshape(-1) for n, v in zip(self.output_names, outs)}
        n_body = len(self.body_names)
        pos_w = o["body_pos_w"].reshape(n_body, 3)
        quat_w = o["body_quat_w"].reshape(n_body, 4)
        lin_b = o["body_lin_vel_b"].reshape(n_body, 3)
        ang_b = o["body_ang_vel_b"].reshape(n_body, 3)
        rot = quat_wxyz_to_matrix(quat_w[self.base])
        return {
            "fps": self.fps,
            "dof_pos": [o["joint_pos"].tolist()],
            "dof_vel": [o["joint_vel"].tolist()],
            "link_body_list": list(self.body_names),
            "keybody_pos_local": [o["body_pos_b"].tolist()],
            "root_pos": [pos_w[self.base].tolist()],
            "root_rot": [quat_w[self.base].tolist()],
            "root_height": float(pos_w[self.base][2]),
            "root_vel": [(rot @ lin_b[self.base]).tolist()],
            "root_angvel": [(rot @ ang_b[self.base]).tolist()],
            "record_meta": {"unix_ms": 0, "default": False, "source": "gmr_frame_pub"},
        }


class Publisher(Node):
    def __init__(self, args):
        super().__init__("gmr_frame_pub")
        self.motion = MotionOnnx(args.motion)
        self.args = args
        self.step = 0
        self.pub = self.create_publisher(String, args.topic, qos_profile_sensor_data)
        self.get_logger().info(
            f"모션={args.motion} bodies={len(self.motion.body_names)} "
            f"frames={self.motion.num_frames} fps={self.motion.fps} -> {args.topic} @ {args.rate}Hz"
        )
        self.create_timer(1.0 / args.rate, self.tick)

    def tick(self):
        payload = self.motion.frame(self.step)
        payload["record_meta"]["unix_ms"] = int(time.time() * 1000)
        self.pub.publish(String(data=repr(payload)))
        self.step += 1
        n = self.motion.num_frames
        if n > 0 and self.step >= n:
            self.step = 0 if self.args.loop else n - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", default="/igris/igris_c_gmt_public/data/onnx/recorded_standstill.onnx")
    p.add_argument("--topic", default="/gmr/teleop/retarget_frame")
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--loop", action="store_true", default=True)
    args = p.parse_args()

    rclpy.init()
    node = Publisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
