"""Arm posture preference taken from the teleoperated robot recordings.

With both hand poses fixed each arm still has one freedom (the swivel of the elbow), and the IK
spends it on a posture preference.  A constant preferred posture left the elbows high and wide
where the operators kept them low: here the preference is looked up instead.  For the hand pose
being asked for, the recorded arm configurations with the most similar hand poses are averaged
(nearest neighbours in the torso frame; the right arm is mirrored onto the left so both arms share
one table).  Source: ``kimyg119/Igris_Teleop_Box_Palletizing`` (full-signals-15hz), commanded joints.
"""
import json
from pathlib import Path

import numpy as np

from .kinematics import ACTIVE, JOINTS, SHOULDERS, UpperBody

TABLE = Path(__file__).with_name('teleop_posture.npz')
MIRROR = np.array([1., -1., -1., 1., -1., -1., 1.])      # right arm joints -> left arm convention (and back)
ORIENTATION_M = .12                                      # a hand direction turned by 1 rad counts like this many metres
NEIGHBOURS = 24
SAMPLES = 16000
SMOOTHING_S = .25                                        # time constant of the looked-up posture


def features(model, q, hands=None):
    """Hand pose of each arm as the table sees it (2 x 9): position from the shoulder, finger direction and
    palm normal, in torso axes, the right arm mirrored.  ``hands`` (two 4x4, pelvis frame) replace the
    hand poses of ``q``; the torso is always that of ``q``."""
    torso, _, own, _ = model.arm_frames(q)
    inverse = np.linalg.inv(torso)
    rows = []
    for side, hand in enumerate(own if hands is None else hands):
        local = inverse @ hand
        position = local[:3, 3] - SHOULDERS[side]
        fingers = -local[:3, 2]; palm = -local[:3, 1] if side == 0 else local[:3, 1]
        row = np.r_[position, ORIENTATION_M * fingers, ORIENTATION_M * palm]
        if side == 1:
            row[[1, 4, 7]] *= -1.                            # mirror about the sagittal plane
        rows.append(row)
    return np.array(rows)


def build(teleop_root, target=TABLE, episodes=None, samples=SAMPLES, seed=0):
    """Write the lookup table from teleoperated episodes (default: every episode with a 'place' task)."""
    import pyarrow.parquet as pq
    from . import teleop
    root = Path(teleop_root); model = UpperBody.from_mjcf()
    if episodes is None:
        rows = [row for path in sorted((root / 'meta/episodes').rglob('*.parquet'))
                for row in pq.read_table(path, columns=['episode_index', 'tasks', 'length']).to_pylist()]
        episodes = [row['episode_index'] for row in rows if 'place' in row['tasks'][0] and row['length'] > 300]
    keys, joints = [], []
    for episode in episodes:
        for q in teleop.load_episode(root, episode)['action'][::2]:
            both = features(model, q)
            keys.extend(both); joints.extend((q[15:22], q[22:29] * MIRROR))
    keys = np.array(keys); joints = np.array(joints)
    keep = np.random.default_rng(seed).choice(len(keys), min(samples, len(keys)), replace=False)
    np.savez_compressed(target, keys=keys[keep], joints=joints[keep], episodes=np.array(episodes),
                        meta=json.dumps(dict(source=str(root), frames=int(len(keys)), kept=int(len(keep)))))
    return target


class TeleopPosture:
    """Looks up the preferred arm joints for the hand poses being commanded."""

    def __init__(self, path=TABLE, neighbours=NEIGHBOURS):
        data = np.load(path)
        self.keys, self.joints, self.neighbours = data['keys'], data['joints'], int(neighbours)
        self.episodes = data['episodes']
        self.squares = np.sum(self.keys ** 2, axis=1)
        self.state = None

    def lookup(self, key):
        """Mean recorded arm joints (left convention) of the nearest hand poses."""
        distance = self.squares - 2. * (self.keys @ key)
        nearest = np.argpartition(distance, self.neighbours)[:self.neighbours]
        return self.joints[nearest].mean(axis=0)

    def reference(self, model, q, left, right, dt=0.):
        """Preferred arm joints (14, SDK order: left then right) for 4x4 pelvis-frame hand targets.

        ``dt`` > 0 low-passes the result over ``SMOOTHING_S`` so that a change of neighbours cannot jerk the arm.
        """
        keys = features(model, q, (left, right))
        wanted = np.r_[self.lookup(keys[0]), self.lookup(keys[1]) * MIRROR]
        if self.state is None or dt <= 0.:
            self.state = wanted
        else:
            self.state = self.state + (1. - np.exp(-dt / SMOOTHING_S)) * (wanted - self.state)
        return self.state.copy()
