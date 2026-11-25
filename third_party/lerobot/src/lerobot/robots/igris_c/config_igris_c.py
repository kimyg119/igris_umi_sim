# [wjson] Igris 코드 추가 — IGRIS-C 로봇 설정
#
# IGRIS-C(31 DOF 휴머노이드) 를 LeRobot 로봇으로 쓰기 위한 설정.
# 관측/제어는 DDS(cyclonedds) 로 로봇 NUC 과 주고받는다.

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.gopro import GoProCameraConfig  # noqa: F401 - register CLI camera type
from lerobot.cameras.ros2 import Ros2CameraConfig  # noqa: F401 - register CLI camera type
from lerobot.utils.ros2_sensor import Ros2SensorConfig

from ..config import RobotConfig
from .tactile import Ros2TactileConfig, SerialTactileConfig, validate_tactile_sources


@RobotConfig.register_subclass("igris_c")
@dataclass
class IgrisCConfig(RobotConfig):
    # DDS 네트워크 (제어 발행용).
    #  관측 수신 쪽 설정(랜카드·피어 주소)은 igris_io.py 가 소유한다.
    namespace: str = "igris_c_IG17"  # 호기 네임스페이스
    domain: int = 0

    # Legacy proprietary DDS image channels are opt-in. Use cameras/tactile ROS2 YAMLs.
    cam_channels: tuple = ()

    # Camera backends (ROS2 in the recording/inference YAMLs).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # A topic selects calibrated ROS2 pressure; a port selects legacy serial input.
    tactile: dict[str, Ros2TactileConfig | SerialTactileConfig] = field(default_factory=dict)
    # Existing camera/tactile configs name features; only this bundle is subscribed.
    ros2_bundle: Ros2SensorConfig | None = None

    # tactile 채널 (sensor/* ImageFrame 8×32 32FC1, 100 Hz).
    # 각 채널은 observation.<이름>_raw (float32 256) + observation.images.<이름>_heatmap 이 된다.
    tactile_channels: tuple = ()

    # Separate torque arrays; observation.state/action remain position-only.
    record_joint_torque: bool = False
    lowstate_timeout_s: float = 0.5

    # --- 안전장치 -------------------------------------------------------
    # True 여야 send_action 이 실제로 로봇에 관절 명령을 발행한다.
    # 수집(record)은 사람이 마스터암으로 조작하므로 False 로 둔다.
    # 정책 추론(rollout)으로 로봇을 직접 움직일 때만 켠다.
    allow_control: bool = False

    # 저수준 제어 게인 (검증 전까지 0 = 토크 없음)
    default_kp: float = 0.0
    default_kd: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.lowstate_timeout_s) or self.lowstate_timeout_s <= 0:
            raise ValueError("lowstate_timeout_s must be finite and positive")
        if self.record_joint_torque and self.type == "igris_c_umi":
            raise ValueError("record_joint_torque requires a robot LowState subscription")
        validate_tactile_sources(self.tactile)
        if self.ros2_bundle is not None and (self.cam_channels or self.tactile_channels):
            raise ValueError("ros2_bundle cannot mix with legacy DDS sensor channels")
        dds_names = {
            channel.removeprefix("rcv_") if channel in ("rcv_tactile_left", "rcv_tactile_right") else channel
            for channel in self.tactile_channels
        }
        for name in self.tactile:
            if not name.isidentifier():
                raise ValueError("Tactile names must be identifiers, e.g. tactile_left")
            if name in dds_names:
                raise ValueError(f"Local and DDS tactile feature collision: {name}")


@RobotConfig.register_subclass("igris_c_upper")
@dataclass
class IgrisCUpperConfig(IgrisCConfig):
    """MasterArm body command subset: left/right arms, DDS indices 15–28 (PJS)."""
