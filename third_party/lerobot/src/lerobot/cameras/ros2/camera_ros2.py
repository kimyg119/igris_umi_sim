"""sensor_msgs/CompressedImage JPEG input for recording, inference and monitoring."""

import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.ros2_sensor import Ros2Subscription

from ..camera import Camera
from .configuration_ros2 import Ros2CameraConfig


class Ros2Camera(Camera):
    def __init__(self, config: Ros2CameraConfig):
        super().__init__(config)
        self.config = config
        self._rx = Ros2Subscription(config, "CompressedImage", self._decode)
        self.frame_lock = threading.Lock()
        self.frame_history: deque = deque(maxlen=1)
        self.latest_frame = self.latest_timestamp = None

    def _decode(self, message: Any) -> NDArray[Any]:
        if "jpeg" not in message.format.lower() and message.format.lower() != "jpg":
            raise ValueError(f"Expected JPEG CompressedImage, got {message.format!r}")
        if not message.data:
            raise ValueError("Empty JPEG payload")
        frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"JPEG dimensions must be {self.width}x{self.height}, got "
                f"{None if frame is None else frame.shape}"
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.frame_lock:
            self.latest_frame, self.latest_timestamp = rgb, time.perf_counter()
            self.frame_history.append((self.latest_timestamp, rgb))
        return rgb

    @property
    def is_connected(self) -> bool:
        return self._rx.is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []  # Explicit YAML topic selection; no USB enumeration.

    def connect(self, warmup: bool = True) -> None:
        self._rx.connect()

    def async_read_with_timestamp(
        self, timeout_ms: float = 200, *, diagnostics: dict[str, Any] | None = None
    ) -> tuple[NDArray[Any], float]:
        if diagnostics is None:
            frame, stamp = self._rx.get(timeout_ms / 1000, new=True)
        else:
            frame, stamp = self._rx.get(timeout_ms / 1000, new=True, diagnostics=diagnostics)
            diagnostics["kind"] = "camera"
        return frame.copy(), stamp

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self.async_read_with_timestamp(timeout_ms)[0]

    def read(self) -> NDArray[Any]:
        return self.async_read()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self._rx.get(max_age_s=max_age_ms / 1000)[0].copy()

    def disconnect(self) -> None:
        self._rx.disconnect()
        with self.frame_lock:
            self.latest_frame = self.latest_timestamp = None
            self.frame_history.clear()
