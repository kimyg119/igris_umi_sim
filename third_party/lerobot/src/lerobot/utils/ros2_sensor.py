"""Read-only ROS2 sensor subscriptions with bounded queues and host freshness checks."""

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _rclpy_available

if TYPE_CHECKING or _rclpy_available:
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions


@dataclass(kw_only=True)
class Ros2SensorConfig:
    topic: str
    domain_id: int | None = None  # None uses ROS_DOMAIN_ID (ROS default: 0).
    timeout_s: float = 0.5
    startup_timeout_s: float = 10.0
    bundle_time_offsets_ms: dict[str, float] = field(default_factory=dict)
    bundle_message_type: str = "five_sensor_msgs/msg/CameraTactileBundle"

    def __post_init__(self):
        if self.bundle_message_type not in (
            "five_sensor_msgs/msg/CameraTactileBundle", "five_sensor_msgs/msg/CameraTactileBundle6"
        ):
            raise ValueError("Unsupported camera/tactile bundle message type")
        if not self.topic.startswith("/") or self.topic == "/" or any(c.isspace() for c in self.topic):
            raise ValueError("ROS2 sensor topic must be an absolute, nonempty topic name")
        if self.domain_id is not None and (type(self.domain_id) is not int or not 0 <= self.domain_id <= 232):
            raise ValueError("domain_id must be None or an integer in [0, 232]")
        for name in ("timeout_s", "startup_timeout_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.bundle_time_offsets_ms, dict) or any(
            not isinstance(name, str) or isinstance(value, bool)
            or not isinstance(value, int | float) or not math.isfinite(value) or abs(value) >= 400
            for name, value in self.bundle_time_offsets_ms.items()
        ):
            raise ValueError("bundle_time_offsets_ms must map sensor names to finite offsets below 400 ms")


class Ros2Subscription:
    """One executor owner per input; closing it never shuts down another ROS context."""

    def __init__(self, config, message_name, decode, *, reliable=False):
        self.config, self.message_name, self.decode = config, message_name, decode
        self.reliable = reliable
        self._condition = threading.Condition()
        self._context = self._node = self._executor = self._thread = None
        self._stop = threading.Event()
        self._latest = self._error = None
        self._sequence = self._consumed = 0
        self._received = 0.0
        self.source_stamp_ns = 0
        self._receive_events = deque(maxlen=128)

    @property
    def is_connected(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def connect(self):
        if self._thread is not None:
            raise RuntimeError(f"{self.config.topic}: already connected; disconnect first")
        if not _rclpy_available:
            raise ImportError(
                "ROS2 sensors require rclpy and sensor_msgs from a sourced ROS2 installation "
                "matching this Python ABI (LeRobot: Python 3.12; ROS2 Jazzy). "
                "For Ubuntu 22.04, install with `pixi install --manifest-path ros2/pixi.toml` "
                "from igris-lerobot, then use port_visualizer.sh or ros2/run.sh. "
                "See examples/igris/ROS2_SENSORS.md; rclpy is not a PyPI extra."
            )
        from rosidl_runtime_py.utilities import get_message

        message_type = get_message(
            self.message_name if "/" in self.message_name else f"sensor_msgs/msg/{self.message_name}"
        )

        self._stop.clear()
        self._latest = self._error = None
        self._sequence = self._consumed = self.source_stamp_ns = 0
        self._receive_events.clear()
        try:
            self._context = Context()
            rclpy.init(
                args=[],
                context=self._context,
                domain_id=self.config.domain_id,
                signal_handler_options=SignalHandlerOptions.NO,
            )
            self._node = rclpy.create_node(
                "lerobot_sensor_" + uuid.uuid4().hex, context=self._context, use_global_arguments=False
            )
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE if self.reliable else ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._node.create_subscription(
                message_type, self.config.topic, self._callback, qos
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name=self._node.get_name(), daemon=True)
            self._thread.start()
            self.get(self.config.startup_timeout_s)
        except BaseException:
            self.disconnect()
            raise

    def _spin(self):
        try:
            while not self._stop.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._latest = None
                self._condition.notify_all()

    def _callback(self, message):
        received = time.perf_counter()
        try:
            stamp = message.header.stamp
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ValueError("Invalid ROS header timestamp")
            source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if source_ns <= 0:
                raise ValueError("ROS sensor messages require a nonzero header timestamp")
            # Repeated/out-of-order samples must not refresh freshness or apply the EMA twice.
            if source_ns <= self.source_stamp_ns:
                return
            data = self.decode(message)
            with self._condition:
                self._latest, self._received = data, received
                self.source_stamp_ns = source_ns
                self._sequence += 1
                self._receive_events.append(received)
                self._error = None
                self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._latest, self._error = None, exc
                self._condition.notify_all()

    def get(self, wait_s=0.0, *, new=False, max_age_s=None, diagnostics=None):
        """Return a sample and receiver perf_counter time; never compare remote clocks."""
        max_age_s = self.config.timeout_s if max_age_s is None else min(max_age_s, self.config.timeout_s)
        deadline = time.perf_counter() + wait_s
        with self._condition:
            while True:
                if self._stop.is_set():
                    raise RuntimeError(f"{self.config.topic}: disconnected")
                if self._error is not None:
                    raise RuntimeError(
                        f"{self.config.topic}: invalid sensor input: {self._error}"
                    ) from self._error
                if (
                    self._latest is not None
                    and time.perf_counter() - self._received <= max_age_s
                    and (not new or self._sequence > self._consumed)
                ):
                    if new:
                        self._consumed = self._sequence
                    if diagnostics is not None:
                        # The metadata belongs to the exact sample returned under this lock.
                        now = time.perf_counter()
                        diagnostics.clear()
                        diagnostics.update(
                            frame_id=str(self.source_stamp_ns),
                            source_stamp_ns=self.source_stamp_ns,
                            received_at=self._received,
                            receive_count=self._sequence,
                            receive_events=[t for t in self._receive_events if now - t <= 2.0],
                            source=self.config.topic,
                        )
                    return self._latest, self._received
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"{self.config.topic}: no fresh ROS2 sample (max age {max_age_s}s)")
                self._condition.wait(remaining)

    def disconnect(self):
        self._stop.set()
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise TimeoutError(f"{self.config.topic}: ROS2 reader did not stop")
            self._thread = None
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._context is not None:
            self._context.try_shutdown()
            self._context = None
