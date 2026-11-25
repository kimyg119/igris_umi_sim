# [wjson] Igris 코드 추가 — IGRIS-C 텔레오퍼레이터 설정
#
# IGRIS 는 마스터암→로봇 제어가 로봇 내부(DDS)에서 이미 이뤄진다.
# 그래서 이 텔레오퍼레이터는 조작 장치를 직접 구동하지 않고,
# 로봇에 전달되는 최종 관절 명령(rt/lowcmd)을 읽어 액션으로 보고한다.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("igris_c")
@dataclass
class IgrisCTeleopConfig(TeleoperatorConfig):
    namespace: str = "igris_c_IG17"
    domain: int = 0

    # lowcmd 가 안 올 때 멈출지 여부.
    #  True(기본): 액션이 전부 0 인 데이터셋이 쌓이는 걸 막는다.
    #  False: 텔레오퍼레이션 없이 파이프라인만 점검할 때 (액션은 0 이 된다).
    require_lowcmd: bool = True


@TeleoperatorConfig.register_subclass("igris_c_upper")
@dataclass
class IgrisCUpperTeleopConfig(IgrisCTeleopConfig):
    """Read only MasterArm's 14 arm targets from the 31-slot LowCmd."""
