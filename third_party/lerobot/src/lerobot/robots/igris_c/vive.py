"""Vive standing or Quest robot-base poses, preserved without coordinate conversion.

The socket belongs to the recording thread.
"""
import json
import logging
import time

import numpy as np

from lerobot.utils.import_utils import require_package

ROLES = ("left_ee", "right_ee")
ROLE_ALIASES = {"LeftUMI": "left_ee", "RightUMI": "right_ee"}
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
ACTION_FEATURES = {f"{role}.{field}": float for role in ROLES for field in POSE_FIELDS}


def _pose(position, quaternion, role: str) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError(f"Invalid pose for {role}")
    if not np.isclose(np.linalg.norm(q), 1, atol=0.01):
        raise ValueError(f"Non-unit quaternion for {role}")
    return np.concatenate((p, q)).astype(np.float32)


def parse_poses(message: dict) -> dict[str, np.ndarray]:
    """Return valid tracked roles; never guess from serial/device order."""
    try:
        if message.get("schema") == "igris_ee_pose_v1":
            if message["ee_frame"] != "robot_base":
                raise ValueError("Quest igris_ee_pose_v1 requires ee_frame=robot_base")
            # This schema has no tracking-validity flags: validate both numeric poses.
            poses = {
                role: _pose(message[side]["pos"], message[side]["quat"], role)
                for side, role in zip(("left", "right"), ROLES, strict=True)
            }
            # Head is optional for the 14-component hand-only action, but when
            # supplied it is validated and retained for HMD-yaw recording.
            if message.get("head"):
                poses["head"] = _pose(message["head"]["pos"], message["head"]["quat"], "head")
            return poses
        if message["schema_version"] != 1 or message["origin"] != "standing":
            raise ValueError("Expected schema_version=1 and origin=standing")
        poses = {}
        seen = set()
        for tracker in message["trackers"].values():
            role = ROLE_ALIASES.get(tracker["role"], tracker["role"])
            if role not in ROLES:
                continue
            if role in seen:
                raise ValueError(f"Duplicate tracker role: {role}")
            seen.add(role)
            if tracker["connected"] is not True or tracker["pose_valid"] is not True or tracker["tracking_state"] != "OK":
                continue
            poses[role] = _pose(tracker["position_m"], tracker["quaternion_xyzw"], role)
        return {role: poses[role] for role in ROLES if role in poses}
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Malformed Vive packet: {exc}") from exc


class ViveSubscriber:
    def __init__(self, endpoint: str, timeout_s: float = 1.0):
        if not endpoint.startswith("tcp://") or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Vive requires a tcp:// endpoint and positive finite timeout_s")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.socket = None
        self._poses = {}
        self._roles = ROLES
        self._held = set()
        self._received = 0.0
        self._stream = None
        self._sequence = -1
        self.coordinate_frame = None

    def connect(self):
        require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
        import zmq

        self.disconnect()
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 256)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def get(self) -> dict[str, np.ndarray]:
        """Hold each last valid pose through tracking loss; only startup waits for data."""
        if self.socket is None:
            raise RuntimeError("Vive subscriber is not connected")
        deadline = time.monotonic() + self.timeout_s
        held = self._held
        while True:
            packet = None
            wait_ms = 0 if all(role in self._poses for role in self._roles) else max(1, int((deadline - time.monotonic()) * 1000))
            if self.socket.poll(wait_ms):
                # Accept JSON alone or [topic, JSON]; drain a bounded batch to the latest.
                for _ in range(256):
                    packet = self.socket.recv_multipart()[-1]
                    if not self.socket.poll(0):
                        break
            if packet is not None:
                try:
                    message = json.loads(packet)
                    poses = parse_poses(message)
                    quest = message.get("schema") == "igris_ee_pose_v1"
                    frame = "robot_base" if quest else "standing"
                    if self.coordinate_frame is not None and frame != self.coordinate_frame:
                        raise ValueError("Pose coordinate frame changed during recording; reconnect to switch sources")
                    stream, sequence = ("igris_ee_pose_v1", message["seq"]) if quest else (
                        message["stream_id"], message["sequence"])
                    if not isinstance(stream, str) or not stream or type(sequence) is not int or sequence < 0:
                        raise ValueError("Invalid Vive stream_id/sequence")
                    if self.coordinate_frame is None:
                        self.coordinate_frame = frame
                        logging.info("Pose source coordinate frame: %s (no coordinate conversion)", frame)
                    # Quest has no stream_id. Match the IK receiver's restart gap.
                    restarted = quest and self._sequence - sequence > 30
                    if stream != self._stream or sequence > self._sequence or restarted:
                        if not self._poses:
                            self._roles = (*ROLES, "head") if "head" in poses else ROLES
                        self._poses.update(poses)
                        self._received = time.monotonic()
                        self._stream, self._sequence = stream, sequence
                        held = set(self._roles) - poses.keys()
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid Vive message from {self.endpoint}: {exc}") from exc
            if all(role in self._poses for role in self._roles):
                if time.monotonic() - self._received > self.timeout_s:
                    held = set(self._roles)
                if held - self._held:
                    logging.warning("Vive tracking unavailable: %s; recording last valid pose(s).",
                                    ", ".join(sorted(held - self._held)))
                if self._held - held:
                    logging.info("Vive tracking recovered: %s.", ", ".join(sorted(self._held - held)))
                self._held = held
                return {role: self._poses[role].copy() for role in self._roles}
            if time.monotonic() >= deadline:
                missing = ", ".join(role for role in self._roles if role not in self._poses)
                raise TimeoutError(f"No initial valid Vive pose for {missing} from {self.endpoint} within {self.timeout_s}s")

    def disconnect(self):
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
        self._poses = {}
        self._roles = ROLES
        self._held = set()
        self._received = 0.0
        self._stream, self._sequence = None, -1
        self.coordinate_frame = None
