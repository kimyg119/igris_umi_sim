"""Joint slack of the vendor simulator model: the angle the SDK commands is not the angle of the link.

Every joint of the vendor MuJoCo model carries a passive slack ("backlash") joint behind its motor, and the
link stands at motor angle + slack angle.  The slack joints have a range of 0.29-0.57 deg, but their stops are
soft: under the weight of the arm the shoulder-pitch and elbow-pitch slack sit at about 1.2 deg all the time.
The IK works with motor angles, so the hands hung 24 mm (median; 16-35 mm) below where they were commanded,
and 16-19 mm off sideways.  At the boxes that stand on other boxes nothing is under the hands; at the lowest
box the right hand came down on the case and stayed on it while the box was taken (6 of 6 runs of episode 7).

``SlackCompensation`` asks the motors for the wanted link angle minus the measured slack.  The slack angles
are simulator ground truth, written by the observer of ``isaac_vendor_room`` at about 30 Hz
(``joint_offsets.bin``); a real robot has no such measurement and would need a calibrated gravity-side offset.
The measurement is low-passed: a slack joint near balance rattles between its two sides, and the motor
target must not follow each flip.
"""
import numpy as np

from .kinematics import ACTIVE, JOINTS

TIME_CONSTANT = .15          # [s] low-pass of the measured slack, as in the earlier bridge compensation
RAMP = 2.                    # [s] over which the correction comes in after it is switched on
FRESH = .15                  # [s] oldest record the reader accepts (the observer writes every 33 ms)
STALE_LIMIT = .5             # [s] without a fresh record before the replay is stopped
ROW = 32 * 8                 # wall time + 31 slack angles, float64
# The gravity side of the slack, for a controller that has no measurement of it (worker ``--slack calibrated``):
# means over five replays of episode 7 (2026-11-24) [deg].  Only the joints that carry the arm's weight sit on one
# side all the time (0.25 deg standard deviation within a run, run means within 0.26 deg of each other); the other
# slack joints swing about zero with the load (shoulder yaw by up to 5 deg) and no constant describes them.
CALIBRATED_DEG = {'l_shoulder_pitch': 1.08, 'l_elbow_pitch': 1.17, 'r_shoulder_pitch': 1.10, 'r_elbow_pitch': 1.14}


def read_slack(path, now):
    """Last complete record of the observer's slack file: (wall time, 31 angles [rad] in SDK joint order),
    or None when the file is missing, empty, not finite, or its last record is not from the last 0.15 s."""
    try:
        with open(path, 'rb') as stream:
            count = stream.seek(0, 2) // ROW
            if not count:
                return None
            stream.seek((count - 1) * ROW)
            row = np.frombuffer(stream.read(ROW), dtype=np.float64)
    except OSError:
        return None
    if len(row) != 32 or not np.isfinite(row).all() or not 0. <= now - row[0] < FRESH:
        return None
    return float(row[0]), row[1:].copy()


def calibrated():
    """31 slack angles [rad] from the calibration constants, in place of a measured record."""
    out = np.zeros(len(JOINTS))
    for name, degrees in CALIBRATED_DEG.items():
        out[JOINTS.index(name)] = np.radians(degrees)
    return out


class SlackCompensation:
    """Motor targets that put the links, not the motors, at the wanted angles."""

    def __init__(self, lower, upper, joints=ACTIVE, time_constant=TIME_CONSTANT, ramp=RAMP, stale_limit=STALE_LIMIT):
        self.lower, self.upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self.joints = list(joints)
        self.time_constant, self.ramp, self.stale_limit = float(time_constant), float(ramp), float(stale_limit)
        self.filtered = np.zeros(len(self.lower))
        self.gain = 0.
        self.age = 0.                                   # time since the last fresh measurement [s]

    def update(self, slack, dt):
        """Take one measurement (31 angles, or None when no fresh record was read) for a tick of ``dt``."""
        self.gain = min(1., self.gain + dt / self.ramp) if self.ramp > 0 else 1.
        if slack is None:
            self.age += dt
            if self.age > self.stale_limit:
                raise RuntimeError(f'No fresh joint slack record for {self.age:.2f} s')
            return
        self.age = 0.
        self.filtered += dt / (self.time_constant + dt) * (np.asarray(slack, dtype=float) - self.filtered)

    @property
    def correction(self):
        """What is taken off the wanted link angles [rad], zero for joints that are not corrected."""
        out = np.zeros(len(self.filtered))
        out[self.joints] = self.gain * self.filtered[self.joints]
        return out

    def motor(self, wanted):
        """Motor angles for wanted link angles, kept inside the joint limits."""
        return np.clip(np.asarray(wanted, dtype=float) - self.correction, self.lower, self.upper)
