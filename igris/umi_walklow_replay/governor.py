"""Joint speed and acceleration limits by slowing the playback, not by leaving the path.

The recorded hands move at human speed: replayed against the clock the arm joints run at their
2 rad/s solver limit (p99), where the teleoperated robot stays below 0.7 rad/s (p99) and
4 rad/s2.  Here the replay advances along the recorded path by a rate between 0 and 1.  Each tick
the IK is asked for the pose a full step ahead; if that step would exceed the joint speed or
acceleration limit it is shortened, and the path position advances by the same share.  The hands
therefore stay on the recorded path and the episode simply takes longer.

The playback rate has a floor.  In front of a target it cannot reach the IK may ask for a step one
way and then the other, tick after tick; the acceleration limit cuts every reversal to almost
nothing, and without the floor the path stood still for as long as the target stayed out of reach
(simulation: 1 s of recording took 24-43 s, the robot held at full stretch meanwhile).  Below the
floor the joints keep their limits and the hands fall behind the path instead.
"""
import numpy as np

from .kinematics import ACTIVE, axis_rotation, rotation_vector

JOINT_SPEED = 1.0            # [rad/s] arms and waist; teleop commands: p99 0.69, p99.9 0.90
JOINT_ACCELERATION = 6.0     # [rad/s2] when speeding up; teleop commands: p99 4.0
RATE_RISE = 2.0              # [1/s] how fast the playback rate may grow back toward 1
RATE_FLOOR = 0.2             # slowest playback rate; the hands leave the path rather than stall it


def blend_pose(a, b, weight):
    """Pose between two 4x4 transforms: straight line in position, shortest turn in orientation."""
    out = np.eye(4)
    out[:3, 3] = (1. - weight) * a[:3, 3] + weight * b[:3, 3]
    turn = rotation_vector(a[:3, :3].T @ b[:3, :3]) * weight
    angle = np.linalg.norm(turn)
    out[:3, :3] = a[:3, :3] if angle < 1e-12 else a[:3, :3] @ axis_rotation(turn / angle, angle)
    return out


class Governor:
    """Path position ``tau`` [s of recorded time] and playback ``rate`` of a replay."""

    def __init__(self, stamps, speed=JOINT_SPEED, acceleration=JOINT_ACCELERATION, rise=RATE_RISE, floor=RATE_FLOOR):
        self.stamps = np.asarray(stamps, dtype=float)
        self.speed, self.acceleration, self.rise = float(speed), float(acceleration), float(rise)
        self.floor = float(floor)
        self.tau = float(self.stamps[0]); self.rate = 0.; self.previous = np.zeros(len(ACTIVE))
        self._ahead = self.tau

    @property
    def done(self):
        return self.tau >= self.stamps[-1] - 1e-9

    def ahead(self, dt):
        """Recorded time a full step ahead; ask the IK for the targets there (``sample``)."""
        self._try = min(1., self.rate + self.rise * dt)
        self._ahead = min(self.stamps[-1], self.tau + self._try * dt)
        return self._ahead

    def sample(self, tau, *series):
        """Each series (poses N x 4 x 4 or values N) at recorded time ``tau``, interpolated."""
        index = int(np.clip(np.searchsorted(self.stamps, tau, side='right') - 1, 0, len(self.stamps) - 2))
        span = self.stamps[index + 1] - self.stamps[index]
        weight = float(np.clip((tau - self.stamps[index]) / span, 0., 1.)) if span > 0 else 0.
        out = [blend_pose(s[index], s[index + 1], weight) if np.ndim(s) == 3 else (1. - weight) * s[index] + weight * s[index + 1]
               for s in series]
        return index, out

    def settle(self, q_now, q_wanted, dt):
        """Shorten the IK's step to the limits, advance the path by the same share (not below the floor),
        return the command."""
        step = (np.asarray(q_wanted, dtype=float) - q_now)[ACTIVE]
        share = 1. / max(1., float(np.max(np.abs(step))) / (self.speed * dt))
        # Speeding up: no joint may gain more than acceleration * dt per tick.  Slowing down is not limited.
        room = self.acceleration * dt * dt
        moving = np.abs(step) > 1e-12
        if moving.any():
            gain = (np.abs(self.previous[moving]) + room) / np.abs(step[moving])
            same = np.sign(step[moving]) == np.sign(self.previous[moving])
            gain = np.where(same | (np.abs(self.previous[moving]) < 1e-12), gain, room / np.abs(step[moving]))
            share = min(share, float(gain.min()))
        share = float(np.clip(share, 0., 1.))
        command = np.array(q_now, dtype=float, copy=True)
        command[ACTIVE] += share * step
        self.previous = share * step
        self.rate = max(self._try * share, min(self.floor, self._try))
        self.tau = min(self.stamps[-1], self.tau + self.rate * dt)
        return command
