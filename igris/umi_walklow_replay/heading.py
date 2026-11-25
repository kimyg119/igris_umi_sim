"""Heading mode of the vendor walking module, used to keep the pelvis from turning.

The vendor walking modules have a heading mode (``setHeadingMode``): the yaw-rate command of the walking policy
becomes 2 x (target heading - measured heading), limited to 1 rad/s, and is dropped below 0.2 rad/s (a heading
error under 0.1 rad).  Nothing in the vendor build switches it on; ``native/heading.cpp`` does, from inside the
simulator's controller process.

The measured heading is that of the IMU, and the IMU sits in the torso, above the three waist joints.  A fixed
target therefore holds the torso, and a replay that turns the waist by 60 deg makes the legs turn the other way
by as much (first trial: the pelvis swung between +48 and -100 deg).  ``torso_heading`` gives the target that
goes with the pelvis standing still: the heading the torso has when the pelvis faces its staged heading and the
waist is where it is now.  The replay sends it every tick.
"""
import math
import os
import struct

import numpy as np

RECORD = struct.Struct('=4d')


def torso_heading(model, q, pelvis_heading):
    """Heading [rad, -pi..pi] of the torso for a level pelvis facing ``pelvis_heading`` and link angles ``q``."""
    rotation = model.torso(q)[:3, :3]
    c, s = math.cos(pelvis_heading), math.sin(pelvis_heading)
    return math.atan2(s * rotation[0, 0] + c * rotation[1, 0], c * rotation[0, 0] - s * rotation[1, 0])


class HeadingTarget:
    """Target file read by ``native/heading.cpp``: sequence, enable, heading, and the sum of the three."""

    def __init__(self, path):
        self.file = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        self.sequence = 0

    def send(self, enable, heading):
        self.sequence += 1
        enable, heading = float(bool(enable)), float(heading)
        os.pwrite(self.file, RECORD.pack(self.sequence, enable, heading, self.sequence + enable + heading), 0)

    def close(self):
        if self.file is not None:
            os.close(self.file); self.file = None


def read_status(path):
    """What the library last did: setter calls, last record applied, and the mode read back from WalkModuleV5."""
    try:
        record = np.fromfile(path, dtype=np.float64)
    except OSError:
        return None
    if len(record) != 4 or not np.isfinite(record).all():
        return None
    return dict(setter_calls=int(record[0]), last_sequence=int(record[1]), enabled=bool(record[2] == 1.),
                target_yaw_rad=float(record[3]))
