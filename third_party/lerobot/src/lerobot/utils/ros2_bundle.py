"""Read one synchronized CameraTactileBundle without opening individual subscriptions."""

import os

from lerobot.cameras.ros2 import Ros2Camera, Ros2CameraConfig
from lerobot.robots.igris_c.ros2_tactile import Ros2Tactile, Ros2TactileConfig
from lerobot.utils.ros2_sensor import Ros2Subscription

SENSOR_NAMES = ("usb_camera_0", "usb_camera_1", "gopro", "tactile_0", "tactile_1")
BUNDLE_FPS = 15


def stamp_ns(header):
    stamp = header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError("Invalid ROS header timestamp")
    value = stamp.sec * 1_000_000_000 + stamp.nanosec
    if value <= 0:
        raise ValueError("Bundle sensors require nonzero timestamps")
    return value


class Ros2Bundle:
    def __init__(self, config, cameras, tactile):
        self.camera_count = 4 if config.bundle_message_type.endswith("/CameraTactileBundle6") else 3
        self.sensor_names = (*SENSOR_NAMES[:3], "realsense", *SENSOR_NAMES[3:]) if self.camera_count == 4 else SENSOR_NAMES
        if len(cameras) != self.camera_count or len(tactile) != 2:
            raise ValueError(f"Bundle requires {self.camera_count} cameras and two tactile pads")
        if set(config.bundle_time_offsets_ms) - set(self.sensor_names):
            raise ValueError("Unknown bundle sensor in time offsets")
        self.offsets_ns = tuple(round(config.bundle_time_offsets_ms.get(n, 0) * 1e6) for n in self.sensor_names)
        self.cameras = [Ros2Camera(c) for c in cameras]
        self.tactile = [Ros2Tactile(c) for c in tactile]
        self.camera_features = dict(zip(self.sensor_names[:self.camera_count], range(self.camera_count), strict=True))
        self.tactile_features = dict(zip(self.sensor_names[self.camera_count:], range(2), strict=True))
        self.sequence, self.stamps = 0, (0,) * len(self.sensor_names)
        # Latest-only observations must not saturate the link with reliable retransmissions.
        self.subscription = Ros2Subscription(
            config, config.bundle_message_type, self.decode, reliable=False
        )

    @classmethod
    def for_observation(cls, config, cameras, tactile):
        """Map existing feature configs to fixed bundle slots, including model subsets."""
        namespace = config.topic.rsplit("/", 1)[0]
        domain = config.domain_id
        if domain is None:
            domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        camera_names = (*SENSOR_NAMES[:3], "realsense") if config.bundle_message_type.endswith("/CameraTactileBundle6") else SENSOR_NAMES[:3]
        camera_configs = [
            Ros2CameraConfig(topic=f"{namespace}/{name}/image_raw/compressed",
                             width=1280 if name == "gopro" else 640,
                             height=720 if name == "gopro" else 480, fps=BUNDLE_FPS)
            for name in camera_names
        ]
        tactile_configs = [Ros2TactileConfig(topic=f"{namespace}/{name}/pressure")
                           for name in SENSOR_NAMES[3:]]
        features = []
        for selected, defaults, expected_type in (
            (cameras, camera_configs, Ros2CameraConfig),
            (tactile, tactile_configs, Ros2TactileConfig),
        ):
            slots = {c.topic: i for i, c in enumerate(defaults)}
            mapping = {}
            for name, sensor in selected.items():
                if not isinstance(sensor, expected_type) or sensor.topic not in slots:
                    raise ValueError(f"{name}: bundle features require a matching ROS2 source topic")
                sensor_domain = sensor.domain_id
                if sensor_domain is None:
                    sensor_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
                if sensor_domain != domain:
                    raise ValueError(f"{name}: bundle and sensor ROS domains differ")
                index = slots[sensor.topic]
                if index in mapping.values():
                    raise ValueError(f"Duplicate bundle source: {sensor.topic}")
                defaults[index] = sensor
                mapping[name] = index
            features.append(mapping)
        if not any(features):
            raise ValueError("Select at least one bundle camera or tactile feature")
        receiver = cls(config, camera_configs, tactile_configs)
        receiver.camera_features, receiver.tactile_features = features
        return receiver

    @property
    def is_connected(self):
        return self.subscription.is_connected

    def read(self, wait_s=0.7, *, include_all_tactile=False, diagnostics=None):
        """Consume one fresh bundle atomically; readiness time is local, not exposure time."""
        received = {}
        if diagnostics is None:
            data, stamp = self.subscription.get(wait_s, new=True)
        else:
            data, stamp = self.subscription.get(wait_s, new=True, diagnostics=received)
        tactile_features = dict(self.tactile_features)
        if include_all_tactile:
            for index, name in enumerate(("tactile_left", "tactile_right")):
                if index not in tactile_features.values():
                    tactile_features[name] = index
        if diagnostics is not None:
            sensors = {}
            for features, offset, kind in (
                (self.camera_features, 0, "camera"), (tactile_features, self.camera_count, "tactile")
            ):
                for name, index in features.items():
                    source_stamp = data["source_stamps_ns"][offset + index]
                    key = name + "_heatmap" if kind == "tactile" else name
                    sensors[key] = {
                        **received,
                        "frame_id": str(source_stamp),
                        "source_stamp_ns": source_stamp,
                        "received_at": stamp,
                        "kind": kind,
                        "source": self.sensor_names[offset + index],
                    }
            diagnostics.clear()
            diagnostics.update(mode="ros2_bundle", bundle_sequence=data["sequence"],
                               received_at=stamp, sensors=sensors)
        return (
            {name: data["images"][index] for name, index in self.camera_features.items()},
            {name: data["pressures"][index] for name, index in tactile_features.items()},
            stamp,
        )

    def decode(self, message):
        if len(message.images) != self.camera_count or len(message.pressures) != 2:
            raise ValueError(f"Expected images[{self.camera_count}] and pressures[2]")
        samples = [*message.images, *message.pressures]
        if tuple(s.header.frame_id for s in samples) != self.sensor_names:
            raise ValueError(f"Bundle sensor order must be {self.sensor_names}")
        stamps = tuple(stamp_ns(s.header) for s in samples)
        if message.reference_sensor not in self.sensor_names:
            raise ValueError("Unknown bundle reference_sensor")
        if stamp_ns(message.header) != stamps[self.sensor_names.index(message.reference_sensor)]:
            raise ValueError("Bundle header timestamp differs from reference sensor")
        aligned = tuple(s + o for s, o in zip(stamps, self.offsets_ns, strict=True))
        aligned_span = max(aligned) - min(aligned)
        if message.span_ns != max(stamps) - min(stamps) or aligned_span > 40_000_000:
            raise ValueError("Bundle raw span must match headers and calibrated span must be at most 40 ms")
        if message.sequence <= self.sequence or any(a <= b for a, b in zip(stamps, self.stamps, strict=True)):
            raise ValueError("Repeated or out-of-order bundle sequence/sensor timestamps")
        images = [c._decode(m) for c, m in zip(self.cameras, message.images, strict=True)]
        previous = [t._visual for t in self.tactile]
        try:
            pressures = [t._decode(m) for t, m in zip(self.tactile, message.pressures, strict=True)]
        except Exception:
            # Reject the whole bundle without applying the left pad's EMA twice on recovery.
            for tactile, visual in zip(self.tactile, previous, strict=True):
                tactile._visual = visual
            raise
        self.sequence, self.stamps = message.sequence, stamps
        return {
            "sequence": message.sequence,
            "span_ns": message.span_ns,
            "aligned_span_ns": aligned_span,
            "source_stamps_ns": stamps,
            "images": images,
            "pressures": pressures,
        }

    def connect(self):
        self.sequence, self.stamps = 0, (0,) * len(self.sensor_names)
        for tactile in self.tactile:
            tactile._visual.fill(0)
        self.subscription.connect()

    def disconnect(self):
        self.subscription.disconnect()
