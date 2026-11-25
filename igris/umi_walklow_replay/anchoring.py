"""Room anchoring that does not chase the pelvis once the arms are out of reach.

Room anchoring keeps the hand targets fixed in the room: whatever the pelvis moves, the targets
move the other way in the pelvis frame.  Along the reach that closes a loop through the walking
controller.  At the lowest box the hands work at 0.94 of the arm length; the IK makes up the rest
by leaning the trunk, the leaning trunk moves the centre of mass forward, and the controller
answers by taking the pelvis back - a few centimetres of sway, or, in the failed runs, a 15 cm
shuffle of the feet within 5 s.  Anchoring then asks for exactly that much more reach.

``ReachHold`` opens the loop where it is ours.  The pelvis displacement is passed on to the
targets as before, except for the part of a retreat (motion away from the hand targets, along the
horizontal line from the pelvis to their middle) that happens while the upper body is strained.
That part is held back, as a displacement fixed in the room: the hands stay that far short of
the room target (``shortfall``).  When the pelvis comes back the held part is taken down first, and
once the strain is gone the rest is given back at a bounded speed.  Sideways motion, height and
heading are always followed.

The held part is a vector, not a distance along the reach: the hands swing through more than 90 deg
between the case and the roll container, and a pelvis that stands 8 cm from where it started must
not count as "retreating" because the hands now point elsewhere (first version, simulation: 14 cm
of false shortfall during a carry, box dropped).

``strain`` is 0 while the arms have reach to spare and 1 when they have none: it rises with the
trunk lean the IK is using toward the hands (forward pitch and roll, as shares of its lean limits)
and with a hand target the IK cannot reach.  Leaning back, as the IK does to carry a box high, is
not strain.
"""
import numpy as np

LEAN_START = .5              # share of the IK's trunk lean limit where following a retreat starts to fade
LEAN_FULL = .8               # share where a retreat is no longer followed at all
ERROR_START = .010           # [m] IK hand position error where following starts to fade
ERROR_FULL = .025            # [m] error where a retreat is no longer followed
RELEASE = .03                # [m/s] how fast a shortfall is given back once the strain is gone


def strain(roll, pitch, position_error, *, roll_limit, forward_pitch_limit, start=LEAN_START, full=LEAN_FULL,
           error_start=ERROR_START, error_full=ERROR_FULL):
    """0..1: how close the upper body is to having no reach left (waist roll, pitch [rad]; IK error [m]).
    ``forward_pitch_limit`` is signed: the waist pitch at which the trunk leans forward as far as the IK may."""
    lean = max(abs(roll) / abs(roll_limit), pitch / forward_pitch_limit)
    return float(np.clip(max((lean - start) / (full - start),
                             (position_error - error_start) / (error_full - error_start)), 0., 1.))


class ReachHold:
    """Share of the pelvis displacement that the hand targets follow; ``shortfall`` [m] is what they do not."""

    def __init__(self, release=RELEASE):
        self.release = float(release)
        self.held = np.zeros(2)          # pelvis displacement that is not passed on [m], staging pelvis axes
        self.offset = np.zeros(2)        # the pelvis starts at the anchor

    @property
    def shortfall(self):
        return float(np.linalg.norm(self.held))

    def update(self, offset, hands, level, dt):
        """Displacement to follow, given the pelvis ``offset`` (x, y, z) and the middle of the hand targets
        ``hands`` (x, y), both in the staging pelvis frame, and the strain ``level``."""
        offset = np.asarray(offset, dtype=float)
        moved = offset[:2] - self.offset; self.offset = offset[:2].copy()
        reach = np.asarray(hands, dtype=float)[:2] - offset[:2]
        distance = np.linalg.norm(reach)
        if distance > 1e-6:
            reach = reach / distance
            along = float(moved @ reach)                 # < 0: the pelvis moves away from the hand targets
            if along < 0:
                self.held += level * along * reach
            else:                                        # coming back: out of what is held along this line first
                self.held += min(along, max(0., -float(self.held @ reach))) * reach
        size = np.linalg.norm(self.held)
        if size > 0:
            self.held *= max(0., size - self.release * (1. - level) * dt) / size
        followed = offset.copy()
        followed[:2] -= self.held
        return followed
