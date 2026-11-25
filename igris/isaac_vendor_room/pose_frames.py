"""Pose rows of a recorded physics run at the times of the video frames.

The physics observer writes a row about 30 times per simulated second, not exactly evenly.  Taking
the next recorded row for each video frame makes the motion uneven: at 10 frames per second the
frames were 65 to 143 ms of simulation apart.  ``frames_at`` gives the pose at the frame time itself,
between the two recorded rows around it.
"""
import numpy as np


def frames_at(rows, column, times):
    """Rows (wall time, simulation time, then x y z qw qx qy qz per body) at ``times`` of clock ``column``.

    Clocks and positions are interpolated linearly, orientations along the shortest turn.  A time outside
    the recording gets the first or the last row.
    """
    rows = np.asarray(rows, dtype=float)
    times = np.asarray(times, dtype=float)
    stamps = rows[:, column]
    after = np.clip(np.searchsorted(stamps, times, side='right'), 1, len(rows) - 1)
    before = after - 1
    span = stamps[after] - stamps[before]
    weight = np.clip((times - stamps[before]) / np.where(span > 0, span, 1.), 0., 1.)[:, None]
    start, end = rows[before], rows[after]
    frames = (1. - weight) * start + weight * end
    poses = frames[:, 2:].reshape(len(times), -1, 7).copy()
    a = start[:, 2:].reshape(len(times), -1, 7)[:, :, 3:]
    b = end[:, 2:].reshape(len(times), -1, 7)[:, :, 3:]
    dot = np.sum(a * b, axis=2, keepdims=True)
    b = np.where(dot < 0, -b, b)                                   # q and -q are the same orientation
    angle = np.arccos(np.clip(np.abs(dot), -1., 1.))
    share = weight[:, :, None]
    near = angle < 1e-6                                            # no turn to speak of: the straight line will do
    sine = np.where(near, 1., np.sin(angle))
    turned = np.where(near, (1. - share) * a + share * b,
                      (np.sin((1. - share) * angle) * a + np.sin(share * angle) * b) / sine)
    poses[:, :, 3:] = turned / np.linalg.norm(turned, axis=2, keepdims=True)
    frames[:, 2:] = poses.reshape(len(times), -1)
    return frames
