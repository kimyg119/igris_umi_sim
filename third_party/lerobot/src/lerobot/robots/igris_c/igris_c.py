# [wjson] Igris 코드 추가 — IGRIS-C 로봇 (LeRobot Robot 구현)
#
# 관측: DDS 로 lowstate(관절 31 + IMU) 와 sensor/*(카메라 JPEG) 를 받는다.
# 제어: SDK LowCmdPublisher 로 관절 목표를 발행한다(안전장치로 기본 비활성).
#
# 수집(record) 시 데이터 흐름:
#   사람 → 마스터암 → (로봇 내부에서 제어) → lowcmd 발행
#   LeRobot: teleop(igris_c).get_action() 이 그 lowcmd 를 액션으로 읽고,
#            robot(igris_c).get_observation() 이 lowstate 를 관측으로 읽는다.
#   robot.send_action() 은 로봇이 이미 그 명령을 실행 중이므로 no-op 이다.

import io
import logging
import time

import cv2
import numpy as np
from PIL import Image

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_igris_c import IgrisCConfig
from .tactile import COLS, ROWS, make_tactile_sensors

logger = logging.getLogger(__name__)

N_JOINTS = 31
UPPER_JOINT_INDICES = tuple(range(15, 29))

# 카메라 해상도 (2026-09-01 실기 JPEG 실측)
CAM_HW = {
    "d435_color": (480, 640),
    "eyes_stereo": (240, 640),
    "rcv_wrist_left": (480, 640),
    "rcv_wrist_right": (480, 640),
}

# DDS 채널명 → 관측 feature 이름. rerun 에는 observation.<이름>,
# 데이터셋에는 observation.images.<이름> 으로 보인다.
CAM_FEATURE = {
    "eyes_stereo": "cam_egoview",
    "rcv_wrist_left": "cam_wrist_left",
    "rcv_wrist_right": "cam_wrist_right",
}

# tactile: 채널명 → feature 접두 (raw 는 observation.<접두>_raw,
# 히트맵은 observation.images.<접두>_heatmap 으로 기록된다)
TACTILE_FEATURE = {
    "rcv_tactile_left": "tactile_left",
    "rcv_tactile_right": "tactile_right",
}
TACTILE_HW = (ROWS, COLS)  # 8×32 taxels, calibrated 0..1
TACTILE_HEATMAP_SCALE = 10  # 80×320 RGB


def joint_names() -> list[str]:
    """SDK python/sdk_gui_client.py JOINT_NAMES order, with explicit left/right names.

    These are index labels only; they do not convert LowCmd motor-space values to PJS.
    """
    return [
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch",
        "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch",
        "right_ankle_pitch", "right_ankle_roll",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch",
        "left_wrist_yaw", "left_wrist_roll", "left_wrist_pitch",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch",
        "right_wrist_yaw", "right_wrist_roll", "right_wrist_pitch",
        "neck_yaw", "neck_pitch",
    ]


def _tactile_heatmap(arr: np.ndarray, colormap: int | None = None) -> np.ndarray:
    """(8,32) or full-frame (12,32) float [0,1] → RGB uint8, ×10 nearest upscaling."""
    import cv2

    if arr.shape not in (TACTILE_HW, (12, 32)) or not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Expected finite calibrated 8x32 or 12x32 tactile values in [0, 1]")
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO if colormap is None else colormap)
    s = TACTILE_HEATMAP_SCALE
    bgr = cv2.resize(bgr, (arr.shape[1] * s, arr.shape[0] * s), interpolation=cv2.INTER_NEAREST)
    return bgr[:, :, ::-1].copy()  # BGR → RGB


class IgrisC(Robot):
    config_class = IgrisCConfig
    name = "igris_c"

    def __init__(self, config: IgrisCConfig):
        from lerobot.utils.ros2_bundle import Ros2Bundle

        super().__init__(config)
        self.config = config
        self.name = config.type
        self.robot_type = config.type
        self._joint_indices = UPPER_JOINT_INDICES if config.type == "igris_c_upper" else tuple(range(N_JOINTS))
        self._rx = None            # igris_io.IgrisReceiver (관측 수신)
        self._lowcmd_pub = None    # SDK LowCmdPublisher (제어 발행)
        self._client = None        # SDK IgrisC_Client (토크/제어모드)
        self._connected = False
        self._telemetry_origin = time.monotonic()
        self._joint_names = [joint_names()[i] for i in self._joint_indices]
        duplicate_tactile = set(config.tactile).intersection(
            TACTILE_FEATURE.get(channel, channel) for channel in config.tactile_channels
        )
        if duplicate_tactile:
            raise ValueError(f"Local and DDS tactile feature collision: {sorted(duplicate_tactile)}")
        self._local_cameras = {}
        self._tactile = make_tactile_sensors(config.tactile)
        reserved = set(self.observation_features)
        duplicates = reserved.intersection(config.cameras)
        if duplicates:
            raise ValueError(f"Camera names collide with IGRIS observations: {sorted(duplicates)}")
        self._local_cameras = make_cameras_from_configs(config.cameras)
        self._sensor_bundle = (
            Ros2Bundle.for_observation(config.ros2_bundle, config.cameras, config.tactile)
            if config.ros2_bundle is not None else None
        )

    # ---- LeRobot 계약: features ----------------------------------------
    @property
    def observation_features(self) -> dict:
        feat: dict = {f"{n}.pos": float for n in self._joint_names}
        if self.config.record_joint_torque:
            feat.update(joint_tau_est_Nm=(len(self._joint_indices),),
                        motor_tau_est_Nm=(len(self._joint_indices),),
                        lowstate_receive_time_s=(1,), lowstate_source_time_s=(1,),
                        lowstate_age_s=(1,))
        for cam in self.config.cam_channels:
            h, w = CAM_HW.get(cam, (480, 640))
            feat[CAM_FEATURE.get(cam, cam)] = (h, w, 3)
        th, tw = TACTILE_HW
        s = TACTILE_HEATMAP_SCALE
        for ch in self.config.tactile_channels:
            base = TACTILE_FEATURE.get(ch, ch)
            feat[f"{base}_raw"] = (th * tw,)  # float32 256, row-major (8×32)
            feat[f"{base}_heatmap"] = (th * s, tw * s, 3)
        for name, sensor in self.config.tactile.items():
            th, tw = sensor.rows, sensor.cols
            feat[f"{name}_raw"] = (th * tw,)
            feat[f"{name}_heatmap"] = (th * s, tw * s, 3)
        for name, camera in self._local_cameras.items():
            feat[name] = (camera.height, camera.width, 3)
        return feat

    @property
    def cameras(self) -> dict:
        """record 가 이미지 기록 스레드 수를 정할 때 len(robot.cameras) 를 쓴다.
        DDS 채널과 로컬 카메라를 함께 포함한다."""
        cams = {CAM_FEATURE.get(c, c): None for c in self.config.cam_channels}
        cams.update(self._local_cameras)
        for ch in self.config.tactile_channels:
            cams[f"{TACTILE_FEATURE.get(ch, ch)}_heatmap"] = None
        cams.update({f"{name}_heatmap": None for name in self.config.tactile})
        return cams

    @property
    def action_features(self) -> dict:
        return {f"{n}.pos": float for n in self._joint_names}

    # ---- 연결 -----------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._telemetry_origin = time.monotonic()
        if self.config.allow_control:
            raise NotImplementedError(
                "IGRIS command publishing is not implemented; use allow_control=False for recording."
            )
        # UMI has no joint subscription; local-camera-only UMI needs no DDS at all.
        if self._joint_indices or self.config.cam_channels or self.config.tactile_channels:
            from igris_io import IgrisReceiver

            self._rx = IgrisReceiver(
                want_cameras=bool(self.config.cam_channels),
                want_tactile=bool(self.config.tactile_channels),
                want_lowstate=bool(self._joint_indices),
            )
            self._rx.start()

        # 관절만 오면 start() 는 끝난다. 하지만 observation_features 에 카메라/tactile 을
        # 선언했으므로, 첫 프레임이 오기 전에 관측을 주면 항목이 빠진다.
        # record 가 데이터셋을 쓸 때 깨지므로 여기서 채워질 때까지 기다린다.
        end = time.time() + 10
        while time.time() < end:
            o = self._rx.get() if self._rx else {"images": {}, "tactile": {}}
            if all(c in o["images"] for c in self.config.cam_channels) and all(
                c in o["tactile"] for c in self.config.tactile_channels
            ):
                break
        else:
            o = self._rx.get()
            missing = [c for c in self.config.cam_channels if c not in o["images"]] + [
                c for c in self.config.tactile_channels if c not in o["tactile"]
            ]
            raise RuntimeError(f"카메라/tactile 첫 프레임이 오지 않음: {missing}")

        if self.config.allow_control:
            self._setup_control()

        try:
            if self._sensor_bundle is not None:
                self._sensor_bundle.connect()
            else:
                for camera in self._local_cameras.values():
                    camera.connect()
                for sensor in self._tactile.values():
                    sensor.connect()
        except BaseException:
            self.disconnect()
            raise

        self._connected = True
        logger.info(f"{self} connected (allow_control={self.config.allow_control})")

    def _setup_control(self) -> None:
        """SDK 로 제어 채널 준비 — allow_control=True 일 때만."""
        import igris_c_sdk as sdk

        ch = sdk.ChannelFactory.Instance()
        ch.Init(self.config.domain, self.config.namespace)
        self._client = sdk.IgrisC_Client()
        self._client.Init()
        self._client.SetTimeout(5.0)
        self._lowcmd_pub = sdk.LowCmdPublisher("rt/lowcmd", sdk.QosProfile.SensorData())
        self._lowcmd_pub.init()
        # TODO(로봇 옆에서 검증 후 활성화): 토크 ON / 제어모드 설정
        #   self._client.SetTorque(sdk.TorqueType.TORQUE_ON, 5000)
        #   self._client.SendControlModeCommand(...)

    # ---- 관측 -----------------------------------------------------------
    def _bundle_observation(self):
        images, pressure, stamp = self._sensor_bundle.read()
        for name, (raw, visual) in pressure.items():
            images[f"{name}_raw"] = raw.reshape(-1)
            images[f"{name}_heatmap"] = _tactile_heatmap(visual, cv2.COLORMAP_VIRIDIS)
        return images, stamp

    def get_observation(self) -> dict:
        out = self._bundle_observation()[0] if self._sensor_bundle is not None else {}
        obs = self._rx.get() if self._rx else {}

        jp = obs.get("joint_pos")  # PJS 관절 위치 (검증 기준 좌표계)
        if jp is not None:
            for name, index in zip(self._joint_names, self._joint_indices, strict=False):
                out[f"{name}.pos"] = float(jp[index])

        if self.config.record_joint_torque:
            received = obs.get("lowstate_receive_monotonic_s")
            source_time = obs.get("lowstate_source_time_s")
            if received is None or source_time is None:
                raise RuntimeError("LowState torque timestamps missing; use the updated igris_io.py")
            age = time.monotonic() - received
            if not np.isfinite(age) or not 0 <= age <= self.config.lowstate_timeout_s:
                raise TimeoutError(f"LowState torque sample is stale: age={age:.3f}s")
            for source, target in (("joint_tau_est", "joint_tau_est_Nm"),
                                   ("motor_tau_est", "motor_tau_est_Nm")):
                values = np.asarray(obs.get(source), dtype=np.float32)
                if values.shape != (N_JOINTS,) or not np.isfinite(values).all():
                    raise ValueError(f"{source} requires 31 finite torque values")
                out[target] = values[list(self._joint_indices)].copy()
            for name, value in (("lowstate_receive_time_s", received - self._telemetry_origin),
                                ("lowstate_source_time_s", source_time), ("lowstate_age_s", age)):
                if not np.isfinite(value):
                    raise ValueError(f"Invalid {name}")
                out[name] = np.array([value], dtype=np.float32)

        for cam, buf in obs.get("images", {}).items():
            if cam not in self.config.cam_channels:
                continue
            out[CAM_FEATURE.get(cam, cam)] = np.array(Image.open(io.BytesIO(buf)).convert("RGB"))

        for ch, arr in obs.get("tactile", {}).items():
            if ch not in self.config.tactile_channels:
                continue
            base = TACTILE_FEATURE.get(ch, ch)
            out[f"{base}_raw"] = arr.reshape(-1).astype(np.float32, copy=False)
            out[f"{base}_heatmap"] = _tactile_heatmap(arr)

        if self._sensor_bundle is None:
            for name, camera in self._local_cameras.items():
                out[name] = camera.async_read()

            for name, sensor in self._tactile.items():
                raw, visual, _ = sensor.get_with_timestamp()
                out[f"{name}_raw"] = raw.reshape(-1)
                out[f"{name}_heatmap"] = _tactile_heatmap(visual, cv2.COLORMAP_VIRIDIS)

        return out

    # ---- 제어 -----------------------------------------------------------
    def send_action(self, action: dict) -> dict:
        """관절 목표를 로봇에 발행.

        allow_control=False 면 실제 발행하지 않는다(no-op).
        수집(record) 중에는 로봇이 이미 마스터암 명령으로 움직이고 있으므로
        LeRobot 이 또 명령을 보내면 충돌한다 → 수집 시엔 False 로 둔다.
        """
        if self.config.allow_control:
            raise NotImplementedError(
                "Direct IGRIS publishing is not implemented; use the guarded ACT WALK_LOW runner."
            )
        if self.name == "igris_c_upper":
            if set(action) != set(self.action_features) or not all(np.isfinite(v) for v in action.values()):
                raise ValueError(
                    "igris_c_upper requires exactly 14 finite named arm positions."
                )
        return action

    # ---- 캘리브레이션: IGRIS 는 로봇/SDK 가 관리 → no-op -----------------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self._sensor_bundle is not None:
            self._sensor_bundle.disconnect()
        else:
            for sensor in self._tactile.values():
                sensor.disconnect()
            for camera in self._local_cameras.values():
                if camera.is_connected:
                    camera.disconnect()
        if self._rx is not None:
            self._rx.stop()
            self._rx = None
        if self._lowcmd_pub is not None:
            self._lowcmd_pub.stop()
            self._lowcmd_pub = None
        self._connected = False
