"""UMI sensor source: images/tactile only; Vive poses are supplied by the recorder."""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import RobotConfig
from .config_igris_c import IgrisCConfig
from .igris_c import IgrisC, _tactile_heatmap
from .vive import ACTION_FEATURES


@RobotConfig.register_subclass("igris_c_umi")
@dataclass
class IgrisCUmiConfig(IgrisCConfig):
    # Sensors come from cameras/tactile configs; legacy proprietary DDS is opt-in.
    cam_channels: tuple = ()
    tactile_channels: tuple = ()
    record_vive: bool = True
    # Quest head.quat -> Z-axis yaw in robot_base, radians in [-pi, pi].
    record_hmd_yaw: bool = False
    # Local host readiness timestamps for cadence/skew diagnostics, not hardware sync.
    record_timing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.record_hmd_yaw and not self.record_vive:
            raise ValueError("record_hmd_yaw requires record_vive=true")
        if self.allow_control:
            raise ValueError("igris_c_umi is a sensor recorder; pose control/IK is not implemented")
        if self.record_timing and (
            self.cam_channels
            or self.tactile_channels
            or any(camera.type not in ("opencv", "gopro", "ros2") for camera in self.cameras.values())
        ):
            raise ValueError("record_timing supports OpenCV/GoPro/ROS2 cameras and serial/ROS2 tactile")


class IgrisCUmi(IgrisC):
    config_class = IgrisCUmiConfig
    name = "igris_c_umi"

    def __init__(self, config: IgrisCUmiConfig):
        super().__init__(config)
        self._joint_names = []
        self._joint_indices = ()
        self._clock_origin = 0.0

    @property
    def observation_features(self) -> dict:
        features = super().observation_features
        features.update(self.action_features)
        if self.config.record_timing:
            features["sensor_bundle_time_s"] = (1,)
            for name in (*self.config.cameras, *self.config.tactile):
                features[f"{name}_receive_time_s"] = (1,)
        return features

    def connect(self, calibrate: bool = True) -> None:
        self._clock_origin = time.perf_counter()
        super().connect(calibrate)

    def get_observation(self) -> dict:
        if not self.config.record_timing:
            return super().get_observation()
        if self._sensor_bundle is not None:
            observation, stamp = self._bundle_observation()
            observation.update({
                f"{name}_receive_time_s": np.array([stamp - self._clock_origin], dtype=np.float32)
                for name in (*self.config.cameras, *self.config.tactile)
            })
            observation["sensor_bundle_time_s"] = np.array([stamp - self._clock_origin], dtype=np.float32)
            return observation
        stamps = {}
        observation = {}
        for name, camera in self._local_cameras.items():
            observation[name], stamps[f"{name}_receive_time_s"] = camera.async_read_with_timestamp()
        for name, sensor in self._tactile.items():
            raw, visual, stamps[f"{name}_receive_time_s"] = sensor.get_with_timestamp()
            observation[f"{name}_raw"] = raw.reshape(-1)
            observation[f"{name}_heatmap"] = _tactile_heatmap(visual, cv2.COLORMAP_VIRIDIS)
        stamps["sensor_bundle_time_s"] = time.perf_counter()
        observation.update(
            {name: np.array([stamp - self._clock_origin], dtype=np.float32) for name, stamp in stamps.items()}
        )
        return observation

    @property
    def action_features(self) -> dict:
        features = dict(ACTION_FEATURES) if self.config.record_vive else {}
        if self.config.record_hmd_yaw:
            features["head.yaw"] = float
        return features

    def send_action(self, action: dict) -> dict:
        if self.config.allow_control:
            raise NotImplementedError("UMI pose control/IK is not implemented")
        if not self.config.record_vive:
            if action:
                raise ValueError("record_vive=false records sensors without actions")
            return {}
        if set(action) != set(self.action_features) or not all(
            np.isscalar(v) and np.isfinite(v) for v in action.values()
        ):
            raise ValueError(f"UMI action requires finite components in {list(self.action_features)}")
        return action
