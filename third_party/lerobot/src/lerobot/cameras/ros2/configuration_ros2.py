"""ROS2 JPEG camera; dimensions describe the publisher's actual decoded image."""

from dataclasses import dataclass

from lerobot.utils.ros2_sensor import Ros2SensorConfig

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("ros2")
@dataclass(kw_only=True)
class Ros2CameraConfig(CameraConfig, Ros2SensorConfig):
    width: int = 640
    height: int = 480
    fps: int = 30
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        Ros2SensorConfig.__post_init__(self)
        for name in ("width", "height", "fps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.color_mode != ColorMode.RGB:
            raise ValueError("ROS2 cameras deliver RGB only")
