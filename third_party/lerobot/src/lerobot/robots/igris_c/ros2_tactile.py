"""Calibrated ROS2 pressure input; baseline/ADC calibration belongs to the publisher."""

from dataclasses import dataclass

import numpy as np

from lerobot.utils.ros2_sensor import Ros2SensorConfig, Ros2Subscription


@dataclass(kw_only=True)
class Ros2TactileConfig(Ros2SensorConfig):
    rows: int = 8
    cols: int = 32
    heatmap_alpha: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        for name in ("rows", "cols"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rows not in (8, 12) or self.cols != 32:
            raise ValueError("Tactile output must be 8x32 or full-frame 12x32")
        if not np.isfinite(self.heatmap_alpha) or not 0 < self.heatmap_alpha <= 1:
            raise ValueError("heatmap_alpha must be in (0, 1]")


class Ros2Tactile:
    def __init__(self, config: Ros2TactileConfig):
        self.config = config
        self._visual = np.zeros((config.rows, config.cols), dtype=np.float32)
        self._rx = Ros2Subscription(config, "Image", self._decode)

    def _decode(self, message):
        c = self.config
        if (
            message.encoding != "32FC1"
            or message.height not in (8, 12)
            or message.width != c.cols
            or message.step < c.cols * 4
            or message.step % 4
            or len(message.data) != message.height * message.step
        ):
            raise ValueError("Expected 8x32 or uncropped 12x32 32FC1 pressure with a valid row stride/payload")
        if message.is_bigendian not in (0, 1):
            raise ValueError("Invalid pressure endianness")
        if message.height < c.rows:
            raise ValueError("Full-frame 12-row checkpoint requires an uncropped 12x32 pressure source")
        dtype = ">f4" if message.is_bigendian else "<f4"
        raw = np.ndarray(
            (message.height, c.cols), dtype=dtype, buffer=bytes(message.data), strides=(message.step, 4)
        ).astype(np.float32, copy=True)
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
            raise ValueError("Pressure must contain finite calibrated values in [0, 1]")
        # Existing policies crop four top rows; full-frame training explicitly requests all twelve.
        raw = raw[-c.rows:, :].copy()
        self._visual = c.heatmap_alpha * raw + (1 - c.heatmap_alpha) * self._visual
        return raw, self._visual.copy()

    def connect(self):
        self._visual.fill(0)
        self._rx.connect()

    def get_with_timestamp(self, *, diagnostics=None):
        if diagnostics is None:
            maps, stamp = self._rx.get()
        else:
            maps, stamp = self._rx.get(diagnostics=diagnostics)
            diagnostics["kind"] = "tactile"
        return *(value.copy() for value in maps), stamp

    def get(self):
        return self.get_with_timestamp()[:2]

    def disconnect(self):
        self._rx.disconnect()
