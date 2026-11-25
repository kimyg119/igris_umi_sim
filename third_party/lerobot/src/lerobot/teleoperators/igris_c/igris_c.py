# [wjson] Igris 코드 추가 — IGRIS-C 텔레오퍼레이터 (LeRobot Teleoperator 구현)
#
# get_action() 은 rt/lowcmd 를 구독해 "지금 로봇에 가해지는 관절 목표"를 돌려준다.
# 이 값이 곧 사람이 마스터암으로 만든 액션이며, 관측(lowstate.joint_state)과
# 같은 31 관절 좌표계라 (관측, 액션) 쌍이 그대로 맞는다.

import logging
import os
import threading
import time

from lerobot.robots.igris_c.igris_c import N_JOINTS, UPPER_JOINT_INDICES, joint_names

from ..teleoperator import Teleoperator
from .config_igris_c import IgrisCTeleopConfig

logger = logging.getLogger(__name__)

class IgrisCTeleop(Teleoperator):
    config_class = IgrisCTeleopConfig
    name = "igris_c"

    def __init__(self, config: IgrisCTeleopConfig):
        super().__init__(config)
        self.config = config
        self.name = config.type
        self._joint_indices = UPPER_JOINT_INDICES if config.type == "igris_c_upper" else tuple(range(N_JOINTS))
        self._joint_names = [joint_names()[i] for i in self._joint_indices]
        self._lock = threading.Lock()
        self._latest = None      # 최신 lowcmd 의 관절 목표 (list[float])
        self._mode_error = None
        self._sub = None         # lowcmd 수신 스레드
        self._run = False
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {f"{n}.pos": float for n in self._joint_names}

    @property
    def feedback_features(self) -> dict:
        return {}   # 마스터암으로 되돌릴 피드백은 쓰지 않는다

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _extract_lowcmd(self, msg) -> None:
        if self.name == "igris_c_upper":
            # Parallel-link groups 3/4 are the wrists. State is always PJS.
            error = None if all(msg.kinematic_modes[i].name == "PJS" for i in (3, 4)) else (
                "igris_c_upper requires PJS wrist commands; MasterArm MS targets do not match joint_state."
            )
            with self._lock:
                self._mode_error = error
            if error:
                return
        try:
            motors = msg.motors() if callable(getattr(msg, "motors", None)) else msg.motors
            vals = [float(motors[i].q() if callable(getattr(motors[i], "q", None)) else motors[i].q)
                    for i in range(N_JOINTS)]
        except Exception:
            return
        with self._lock:
            self._latest = vals

    def connect(self, calibrate: bool = True) -> None:
        """cyclonedds 로 rt/lowcmd 를 직구독한다.

        원래는 SDK LowCmdSubscriber 를 썼지만, SDK whl 의 네이티브 확장이
        glibc 2.38 을 요구해 이 호스트(Ubuntu 22.04, glibc 2.35)에서 로드가
        안 된다. LowCmd 타입은 SDK IDL 생성본(messages)에 있으므로 igris_io
        와 동일한 cyclonedds 직구독으로 같은 데이터를 받는다.
        """
        # record 는 teleop 을 robot 보다 먼저 연결한다. DDS 가 기본 랜카드를
        # 잡지 않도록, 로봇 직결 설정을 여기서 먼저 넣는다(igris_io 와 동일).
        import igris_io

        os.environ["CYCLONEDDS_URI"] = igris_io.URI

        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader, Subscriber
        from cyclonedds.topic import Topic

        from messages.igris_c_msgs import LowCmd

        self._dp = DomainParticipant(self.config.domain)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(5))
        topic = Topic(self._dp, f"{self.config.namespace}/rt/lowcmd", LowCmd, qos=qos)
        reader = DataReader(Subscriber(self._dp), topic, qos=qos)

        self._run = True

        def loop():
            while self._run:
                last = None
                for m in reader.take(N=50):
                    last = m
                if last is not None:
                    self._extract_lowcmd(last)
                time.sleep(0.001)

        self._sub = threading.Thread(target=loop, daemon=True)
        self._sub.start()

        # 첫 명령이 올 때까지 잠깐 대기 (마스터암이 조작 중이어야 값이 온다)
        end = time.time() + 5
        while time.time() < end and self._latest is None:
            time.sleep(0.05)
        if self._mode_error:
            self.disconnect()
            raise ValueError(self._mode_error)
        if self._latest is None and not self.config.require_lowcmd:
            logger.warning("rt/lowcmd 명령이 없습니다 — 액션이 전부 0 으로 기록됩니다.")
        elif self._latest is None:
            # 그냥 진행하면 get_action() 이 0 을 돌려주고, 액션이 전부 0 인
            # 쓸모없는 데이터셋이 조용히 쌓인다. 여기서 멈추는 편이 낫다.
            raise RuntimeError(
                "rt/lowcmd 에서 명령이 오지 않습니다. 마스터암 텔레오퍼레이션이 "
                "실행 중인지 확인하세요. (DDS 연결이 간헐 실패하기도 하니 재실행도 시도)"
            )
        self._connected = True

    def get_action(self) -> dict:
        with self._lock:
            if self._mode_error:
                raise ValueError(self._mode_error)
            vals = self._latest
        if vals is None:
            return {f"{n}.pos": 0.0 for n in self._joint_names}
        return {f"{n}.pos": vals[i] for n, i in zip(self._joint_names, self._joint_indices)}

    def send_feedback(self, feedback: dict) -> None:
        pass   # 사용하지 않음

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self._run = False
        if self._sub is not None:
            self._sub.join(timeout=2)
            self._sub = None
        self._connected = False
