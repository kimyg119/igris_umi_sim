"""Camera inputs for the standard recorder, with no robot or action channel."""

import math
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.cameras.gopro import GoProCameraConfig  # noqa: F401 - CLI registration
from lerobot.cameras.ros2 import Ros2CameraConfig  # noqa: F401 - CLI registration
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.ros2_bundle import Ros2Bundle
from lerobot.utils.ros2_sensor import Ros2SensorConfig

from .config import RobotConfig
from .robot import Robot


@RobotConfig.register_subclass("camera_only")
@dataclass
class CameraOnlyConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    ros2_bundle: Ros2SensorConfig | None = None
    # Measured relative decoder latency. None preserves the legacy latest-frame mode.
    camera_latency_ms: dict[str, float] | None = None
    max_camera_skew_ms: float = 40.0

    def __post_init__(self):
        super().__post_init__()
        if not self.cameras:
            raise ValueError("camera_only requires at least one camera")
        if not math.isfinite(self.max_camera_skew_ms) or not 0 < self.max_camera_skew_ms <= 1000:
            raise ValueError("max_camera_skew_ms must be in (0, 1000]")
        if self.camera_latency_ms is not None:
            if self.ros2_bundle is not None:
                raise ValueError("ros2_bundle cannot mix with camera_latency_ms alignment")
            if set(self.camera_latency_ms) != set(self.cameras):
                raise ValueError("Provide measured camera_latency_ms for every camera")
            if any(not math.isfinite(v) or not 0 <= v <= 1000 for v in self.camera_latency_ms.values()):
                raise ValueError("camera_latency_ms must contain finite values in [0, 1000]")
            if any(c.type not in ("opencv", "gopro", "ros2") for c in self.cameras.values()):
                raise ValueError("Time alignment supports OpenCV, GoPro and ROS2 cameras")


class CameraOnly(Robot):
    config_class = CameraOnlyConfig
    name = "camera_only"

    def __init__(self, config: CameraOnlyConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._sensor_bundle = (
            Ros2Bundle.for_observation(config.ros2_bundle, config.cameras, {})
            if config.ros2_bundle is not None else None
        )
        self._connections = ExitStack()
        self.config = config
        self._last_target = -math.inf
        self._clock_origin = 0.0
        if config.camera_latency_ms is not None:
            seconds = max(config.camera_latency_ms.values()) / 1000 + 0.25
            for name, camera in self.cameras.items():
                camera.frame_history = deque(maxlen=math.ceil(seconds * config.cameras[name].fps) + 2)

    @property
    def observation_features(self) -> dict:
        features = {name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()}
        if self.config.camera_latency_ms is not None:
            for name in self.cameras:
                features[f"{name}_receive_time_s"] = (1,)
                features[f"{name}_estimated_capture_time_s"] = (1,)
        return features

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        if self._sensor_bundle is not None:
            return self._sensor_bundle.is_connected
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self._clock_origin = time.perf_counter()
        self._last_target = -math.inf
        try:
            if self._sensor_bundle is not None:
                self._connections.callback(self._sensor_bundle.disconnect)
                self._sensor_bundle.connect()
            else:
                for camera in self.cameras.values():
                    self._connections.enter_context(camera)
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self) -> dict:
        if self._sensor_bundle is not None:
            return self._sensor_bundle.read()[0]
        if self.config.camera_latency_ms is not None:
            return self._synchronized_observation()
        return {name: cam.async_read(timeout_ms=2000) for name, cam in self.cameras.items()}

    def _synchronized_observation(self) -> dict:
        """Match estimated exposure times, not the times USB/decoder delivery finishes.

        Constant measured offsets cannot remove sensor/codec jitter or replace hardware sync.
        """
        offsets = {name: ms / 1000 for name, ms in self.config.camera_latency_ms.items()}
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline:
            histories = {}
            for name, camera in self.cameras.items():
                with camera.frame_lock:
                    histories[name] = list(camera.frame_history)
            if all(histories.values()):
                now = time.perf_counter()
                if any(now - h[-1][0] > 0.5 for h in histories.values()):
                    raise TimeoutError("Camera stopped delivering fresh frames during alignment")
                target = min(h[-1][0] - offsets[name] for name, h in histories.items())
                covered = all(h[0][0] - offsets[name] <= target for name, h in histories.items())
                if target > self._last_target and covered:
                    selected = {
                        name: min(h, key=lambda entry: abs(entry[0] - offsets[name] - target))
                        for name, h in histories.items()
                    }
                    capture_times = [stamp - offsets[name] for name, (stamp, _) in selected.items()]
                    if max(capture_times) - min(capture_times) <= self.config.max_camera_skew_ms / 1000:
                        self._last_target = target
                        out = {}
                        for name, (stamp, frame) in selected.items():
                            out[name] = frame
                            out[f"{name}_receive_time_s"] = np.array(
                                [stamp - self._clock_origin], dtype=np.float32
                            )
                            out[f"{name}_estimated_capture_time_s"] = np.array(
                                [stamp - offsets[name] - self._clock_origin], dtype=np.float32
                            )
                        return out
            time.sleep(0.001)
        raise TimeoutError("No fresh camera bundle within max_camera_skew_ms; check calibration and streams")

    def send_action(self, action: dict) -> dict:
        if action:
            raise ValueError("camera_only does not accept actions")
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self._connections.close()
