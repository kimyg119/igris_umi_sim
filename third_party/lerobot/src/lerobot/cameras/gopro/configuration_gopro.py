"""GoPro USB webcam configuration (Open GoPro HTTP API)."""

from dataclasses import dataclass
from ipaddress import IPv4Address

from ..configs import CameraConfig


@CameraConfig.register_subclass("gopro")
@dataclass(kw_only=True)
class GoProCameraConfig(CameraConfig):
    ip_address: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    port: int = 554

    def __post_init__(self) -> None:
        IPv4Address(self.ip_address)
        if (self.width, self.height) != (1920, 1080):
            raise ValueError(
                "GoPro USB webcam frames are stored at their native 1920x1080 resolution; "
                "resizing and letterboxing are not supported."
            )
        if self.fps != 30:
            raise ValueError("GoPro webcam supports 30 fps.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("GoPro RTSP server port must be between 1 and 65535.")
