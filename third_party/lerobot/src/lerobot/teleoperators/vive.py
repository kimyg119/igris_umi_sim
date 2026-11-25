"""Record Vive poses as actions; no robot command publisher or lowcmd subscription."""
from dataclasses import dataclass

import numpy as np

from lerobot.robots.igris_c.vive import ACTION_FEATURES, POSE_FIELDS, ROLES, ViveSubscriber

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


@TeleoperatorConfig.register_subclass("vive")
@dataclass
class ViveTeleopConfig(TeleoperatorConfig):
    endpoint: str = "tcp://223.195.39.197:5555"
    timeout_s: float = 1.0
    # lerobot-record mirrors robot.record_hmd_yaw so state/action schemas agree.
    record_hmd_yaw: bool = False


class ViveTeleop(Teleoperator):
    config_class = ViveTeleopConfig
    name = "vive"

    def __init__(self, config: ViveTeleopConfig):
        super().__init__(config)
        self.config = config
        self._rx = ViveSubscriber(config.endpoint, config.timeout_s)
        self._connected = False

    @property
    def action_features(self) -> dict:
        features = dict(ACTION_FEATURES)
        if self.config.record_hmd_yaw:
            features["head.yaw"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        try:
            self._rx.connect()
            self.get_action()
        except BaseException:
            self.disconnect()
            raise
        self._connected = True

    def get_action(self) -> dict:
        poses = self._rx.get()
        action = {f"{role}.{field}": float(poses[role][i])
                  for role in ROLES for i, field in enumerate(POSE_FIELDS)}
        if self.config.record_hmd_yaw:
            if "head" not in poses:
                raise ValueError("HMD yaw recording requires Quest head.quat in ee_frame=robot_base")
            q = poses["head"][3:].astype(np.float64)
            x, y, z, w = q / np.linalg.norm(q)
            # Z-axis yaw in robot_base, matching the IK retargeter's yaw convention.
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            action["head.yaw"] = float(yaw)
        return action

    @staticmethod
    def observation_from_action(action: dict) -> dict:
        """Reconstruct hand observations from this sample without another ZMQ read."""
        return {
            role: np.array([action[f"{role}.{field}"] for field in POSE_FIELDS], dtype=np.float32)
            for role in ROLES
        }

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        self._rx.disconnect()
        self._connected = False
