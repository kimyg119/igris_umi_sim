"""What a policy sees of the robot and what it says to it: three "end effectors" - the chest and the two hands.

All of it comes from the joint angles through the links of the robot (kinematics.UpperBody), numpy only: the
same code makes the training labels, runs next to the policy and runs in the controller.

State (30 values), nothing in it refers to the room or to where the pelvis stands:
  left hand, right hand   pose in the chest (torso link) frame, as position + the first two columns of the rotation
  the pair                the right hand in the frame of the left: how far apart the hands are and how they face
  waist                   the three waist joint angles (yaw, roll, pitch): where the chest is on the pelvis

Action (21 values per step of a chunk), relative to the state at the moment of the observation:
  left hand               target pose expressed in the frame the hand has now (position + two rotation columns):
                          how the hand has to move from where it is, as its own wrist camera sees it
  the pair                the right hand's target in the frame of the left hand's target
  waist                   target waist joint angles minus the angles now

The two hands carry a box by pressing on it from both sides, 27 mm (21-32 mm) closer together than the box lets
them come.  That squeeze is one quantity, and it is given as one: while a box is held the pair is a constant
(286 mm apart in every carry of the data), whatever the left hand does, and an error in the left hand's action
moves both hands and the box with them.  Whether that is better than a relative action of its own for each hand
(``paired=False``) was not shown: after the same 4,000 training steps the gap between the hands while carrying
was predicted 4.2 mm off (p95 6.1) paired and 3.9 mm off (p95 9.5) unpaired - the two hands' errors move together -
and in closed loop 22 of 36 against 19 of 33 boxes were placed (README).

A hand action is the same numbers whichever frame the two poses are given in, as long as it is the same frame
for both: the hand "now" and the target.  Labels are made with both in the room (the simulator knows where the
pelvis is), so they say how the hand moves in space.  The controller composes them with the hand pose in the
pelvis frame, which is all a robot knows by itself: whatever the pelvis does during a chunk is then an error
that the next observation corrects.

The waist is a joint target and not a third pose because that is all it can be: three joints in a row, with
their limits on the joints.  It is relative for the same reason the hands are: the pelvis turns on its feet
(20 degrees and more in a minute of standing), and "turn until the roll container is in front" survives that,
"waist yaw = -60 degrees" does not.
"""
import numpy as np

from .kinematics import JOINTS
from .slack import calibrated

STATE_SIZE = 30
ACTION_SIZE = 21
PARTS = ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21')
STATE_NAMES = ([f'{name}.{part}' for name in ('left_ee_chest', 'right_ee_chest', 'right_ee_in_left') for part in PARTS]
               + ['waist_yaw', 'waist_roll', 'waist_pitch'])
ACTION_NAMES = ([f'{name}.{part}' for name in ('left_ee_relative', 'right_target_in_left_target') for part in PARTS]
                + ['waist_yaw_delta', 'waist_roll_delta', 'waist_pitch_delta'])


def pose9(pose):
    """Position and the first two rotation columns of 4x4 poses (... x 4 x 4 -> ... x 9)."""
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[..., :3, 3], pose[..., :3, 0], pose[..., :3, 1]], axis=-1)


def from9(values):
    """4x4 poses of ``pose9`` values; the two columns are made orthonormal (Gram-Schmidt), the third is their cross."""
    values = np.asarray(values, dtype=float)
    first = values[..., 3:6]; second = values[..., 6:9]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    out = np.zeros(values.shape[:-1] + (4, 4))
    out[..., :3, 0] = first; out[..., :3, 1] = second; out[..., :3, 2] = np.cross(first, second)
    out[..., :3, 3] = values[..., :3]; out[..., 3, 3] = 1.
    return out


def inverse(pose):
    """Inverse of rigid 4x4 transforms."""
    pose = np.asarray(pose, dtype=float)
    out = np.zeros_like(pose)
    turned = np.swapaxes(pose[..., :3, :3], -1, -2)
    out[..., :3, :3] = turned
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', turned, pose[..., :3, 3])
    out[..., 3, 3] = 1.
    return out


def links(q_measured):
    """Link angles a controller believes in: the measured motor angles plus the calibrated joint slack (slack.py)."""
    return np.asarray(q_measured, dtype=float) + calibrated()


def body(model, q_measured):
    """Hands in the pelvis frame (2 x 4 x 4), the chest in the pelvis frame (4 x 4) and the waist angles (3)
    for 31 measured joint angles."""
    q = links(q_measured)
    hands = np.array(model.fk(q))
    return hands, model.torso(q), np.asarray(q_measured, dtype=float)[:3].copy()


def state(hands_pelvis, chest_pelvis, waist):
    """The 30 state values from the hands (... x 2 x 4 x 4) and the chest (... x 4 x 4) in one common frame and the
    waist joint angles (... x 3)."""
    hands = np.asarray(hands_pelvis, dtype=float)
    in_chest = inverse(np.asarray(chest_pelvis, dtype=float))[..., None, :, :] @ hands
    pair = inverse(hands[..., 0, :, :]) @ hands[..., 1, :, :]
    return np.concatenate([pose9(in_chest[..., 0, :, :]), pose9(in_chest[..., 1, :, :]), pose9(pair),
                           np.asarray(waist, dtype=float)], axis=-1)


def relative_actions(hands_now, waist_now, hand_targets, waist_targets, paired=True):
    """Chunk of actions (K x 21) for hand targets (K x 2 x 4 x 4) and waist targets (K x 3), relative to the hands
    (2 x 4 x 4, same frame as the targets) and the waist angles (3) at the moment of the observation.
    ``paired=False``: the right hand like the left, each relative to its own present pose."""
    hand_targets = np.asarray(hand_targets, dtype=float); hands_now = np.asarray(hands_now, dtype=float)
    left = inverse(hands_now[0])[None] @ hand_targets[:, 0]
    right = inverse(hand_targets[:, 0]) @ hand_targets[:, 1] if paired else inverse(hands_now[1])[None] @ hand_targets[:, 1]
    return np.concatenate([pose9(left), pose9(right),
                           np.asarray(waist_targets, dtype=float) - np.asarray(waist_now, dtype=float)], axis=-1)


def absolute_targets(hands_now, waist_now, actions, paired=True):
    """The inverse of ``relative_actions``: hand targets (K x 2 x 4 x 4) in the frame ``hands_now`` is given in,
    and waist joint targets (K x 3)."""
    actions = np.asarray(actions, dtype=float)
    hands_now = np.asarray(hands_now, dtype=float)
    left = hands_now[0] @ from9(actions[:, :9])
    right = left @ from9(actions[:, 9:18]) if paired else hands_now[1] @ from9(actions[:, 9:18])
    return np.stack([left, right], axis=1), np.asarray(waist_now, dtype=float) + actions[:, 18:21]


def pair_targets(actions):
    """Hand targets (K x 2 x 4 x 4) of actions that are not relative to anything: the left hand's target as it
    is, the right hand's in the frame of the left's (the absolute variant with paired hands)."""
    actions = np.asarray(actions, dtype=float)
    left = from9(actions[:, :9])
    return np.stack([left, left @ from9(actions[:, 9:18])], axis=1)


WAIST = [JOINTS.index(name) for name in ('waist_yaw', 'waist_roll', 'waist_pitch')]
