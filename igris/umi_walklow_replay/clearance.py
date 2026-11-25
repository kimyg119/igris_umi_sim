"""Keep the robot's hands off the surfaces the boxes stand on.

The recorded palm passes close to the case when the operator takes the lowest box: replayed with the robot's
rigid hand (the contact pad reaches 10 cm past the palm point), the right pad is left 4 mm of air 0.8 s before
the grasp of episode 7.  At the boxes that stand on other boxes nothing is under the hands, so only the lowest
box shows it.  With the hands running true to their targets (slack.py) the pad still brushed the case in 5 of 6
simulated runs.

``clear_supports`` raises a hand target wherever its contact pad would come nearer than ``margin`` to a known
support surface (the case top, the loading surface of the roll container).  Only the height changes, by the
missing amount, and the lift comes and goes at a bounded speed.  The surfaces are scene knowledge: a policy
that outputs hand poses does not have them, an executor on the robot would need them from perception.
"""
import numpy as np

MARGIN = .02                 # [m] air kept between the contact pad and a support surface
LIFT_SPEED = .05             # [m/s] how fast the added height may change
FOOTPRINT_PAD = .01          # [m] a support counts this far beyond its edges


def cone_envelope(deficit, step):
    """Smallest lift that covers ``deficit`` at every sample and changes by at most ``step`` per sample."""
    lift = np.asarray(deficit, dtype=float).copy()
    for i in range(1, len(lift)):                    # rising no faster than the limit ...
        lift[i] = max(lift[i], lift[i - 1] - step)
    for i in range(len(lift) - 2, -1, -1):           # ... and falling no faster
        lift[i] = max(lift[i], lift[i + 1] - step)
    return lift


def clear_supports(targets, corners, height_offset, surfaces, dt, margin=MARGIN, speed=LIFT_SPEED):
    """Raised copies of EE ``targets`` (N x 4 x 4) and the lift per sample [m].

    ``corners`` are points of the hand's contact pad in the EE frame; ``height_offset`` is what the replay adds
    to the target heights per sample to get room heights; ``surfaces`` are dicts with ``centre`` (x, y), ``yaw``,
    ``half`` (half sizes along the surface's own axes) and ``top`` (room height), in the frame of the targets."""
    targets = np.asarray(targets, dtype=float)
    points = np.einsum('nij,kj->nki', targets[:, :3, :3], np.asarray(corners, dtype=float)) + targets[:, None, :3, 3]
    heights = points[:, :, 2] + np.asarray(height_offset, dtype=float)[:, None]
    deficit = np.zeros(len(targets))
    for surface in surfaces:
        c, s = np.cos(surface['yaw']), np.sin(surface['yaw'])
        offset = points[:, :, :2] - np.asarray(surface['centre'], dtype=float)
        along, across = c * offset[:, :, 0] + s * offset[:, :, 1], -s * offset[:, :, 0] + c * offset[:, :, 1]
        over = (np.abs(along) <= surface['half'][0] + FOOTPRINT_PAD) & (np.abs(across) <= surface['half'][1] + FOOTPRINT_PAD)
        missing = np.where(over, surface['top'] + margin - heights, 0.).max(axis=1)
        deficit = np.maximum(deficit, missing)
    lift = cone_envelope(np.maximum(deficit, 0.), speed * dt)
    raised = targets.copy()
    raised[:, 2, 3] += lift
    return raised, lift
