"""Constant calibration of the two recorded hand frames, measured on the boxes they hold.

The recording gives each hand a pose whose axes should be the robot EE axes (thumb +x, palm axis
+y pointing from the right palm to the left palm, fingers -z) and whose origin should be the palm
centre.  The two devices do not agree: seen from the right hand, the left hand is turned about
the palm axis by 28 deg (std 2.8 deg over 22 000 held frames of 34 episodes, 1.6 deg between
episodes), and while a box rests level on a surface the left origin is 4.9 +- 0.9 cm lower than
the right one.  Replayed as recorded, the left robot hand comes down pointing 25-30 deg further
down and 6-9 cm lower than the right hand and swings level only as the grasp closes.

What a held box says about the hands, and nothing more is used here:
- While the box rests level, a palm lying on its end face has a horizontal normal.  The normal
  is therefore the direction of the hand frame that stays horizontal over all such frames.
- Both hands hold the same rigid box, so their corrected frames agree in the turn about the
  palm axis.  Which hand is right is not observable; the right hand is taken as the reference
  because its approach looks natural in the recording and in the replay (user, 2026-11-23).
- The line between the two palm centres is the box axis, i.e. the common palm axis.  The part of
  the recorded line across it is a constant error of the origins; which origin carries it is not
  observable, so each hand takes half and the midpoint of the pair stays where it was recorded.

``measure`` returns the constants that ``retarget.HAND_CALIBRATION`` holds.  Only numpy is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .episode import continuous_quaternions
from .kinematics import axis_rotation, matrix_quat, quat_matrix

UP = np.array([0., 0., 1.])
Y = np.array([0., 1., 0.])
# Measured with ``measure`` on all 34 episodes of the box-palletizing recording (2026-11-23, repaired tracking):
# palm axis in each recorded frame, turn of the left hand about it, and the origin shift in the corrected axes.
CALIBRATION = {'left': dict(normal=[-.0030, .9208, .3901], roll_deg=-28.16, offset_m=[.0309, 0., .0055]),
               'right': dict(normal=[-.2024, .9198, -.3360], roll_deg=0., offset_m=[-.0309, 0., -.0055])}


def normal_rotation(normal):
    """Rotation C with ``C @ y = normal``: turns a recorded frame so that its +y is the palm axis."""
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    axis = np.cross(Y, normal)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3)
    return axis_rotation(axis / sine, np.arctan2(sine, normal[1]))


def correction(entry):
    """Rotation C (``R_corrected = R_recorded @ C``) of one hand's calibration entry."""
    return normal_rotation(entry['normal']) @ axis_rotation(Y, np.radians(float(entry.get('roll_deg', 0.))))


def apply(episode, calibration=CALIBRATION):
    """The episode with both recorded hand poses (xyz + xyzw rows) corrected by the constants."""
    out = dict(episode)
    for side in ('left', 'right'):
        pose = np.array(episode[side], dtype=float)
        entry = calibration[side]
        rotations = np.array([quat_matrix(row[[6, 3, 4, 5]]) for row in pose]) @ correction(entry)
        pose[:, :3] += np.einsum('nij,j->ni', rotations, np.asarray(entry['offset_m'], dtype=float))
        pose[:, 3:] = continuous_quaternions(np.array([matrix_quat(r) for r in rotations])[:, [1, 2, 3, 0]])
        out[side] = pose
    return out


def frames(segments, count):
    """Indices where the held box rests on a surface, and where it is held at all."""
    rest, held = [], []
    for segment in segments:
        first, last = int(segment['pick_index']), int(segment['place_index'])
        rest += [i for i in (*range(first + 1, first + 5), *range(last - 5, last - 1)) if 0 <= i < count and first < i < last]
        held += list(range(first + 3, max(first + 3, last - 3)))
    return np.array(rest, dtype=int), np.array(held, dtype=int)


def measure(episodes):
    """Calibration constants from ``(left N x 4 x 4, right N x 4 x 4, carry segments)`` per episode."""
    left, right, rest, held, start = [], [], [], [], 0
    for one_left, one_right, segments in episodes:
        r, h = frames(segments, len(one_left))
        rest.append(r + start); held.append(h + start); start += len(one_left)
        left.append(np.asarray(one_left, dtype=float)); right.append(np.asarray(one_right, dtype=float))
    left, right = np.concatenate(left), np.concatenate(right)
    rest, held = np.concatenate(rest), np.concatenate(held)
    if len(rest) < 8 or len(held) < 8:
        raise ValueError('Too few held frames to calibrate the hands')
    between = left[held, :3, 3] - right[held, :3, 3]               # right palm -> left palm

    def normal(poses):
        # World up seen in the hand frame while the box rests level: the palm axis is across all of them.
        seen = np.einsum('nji,j->ni', poses[rest, :3, :3], UP)
        _, values, basis = np.linalg.svd(seen, full_matrices=False)
        axis = basis[-1]
        if np.mean(np.einsum('nij,j,ni->n', poses[held, :3, :3], axis, between)) < 0:
            axis = -axis
        return axis, values

    out, report = {}, {}
    turned = {}
    for side, poses in (('left', left), ('right', right)):
        axis, values = normal(poses)
        out[side] = dict(normal=axis.tolist(), roll_deg=0., offset_m=[0., 0., 0.])
        turned[side] = poses[:, :3, :3] @ normal_rotation(axis)
        report[side + '_normal_singular_values'] = values.tolist()
        report[side + '_normal_elevation_at_rest_deg'] = float(np.degrees(np.arcsin(
            np.clip(np.einsum('nij,j->ni', poses[rest, :3, :3], axis)[:, 2], -1, 1))).std())
    # Turn of the left thumb axis about the palm axis, seen from the right hand.
    thumb = np.einsum('nji,nj->ni', turned['right'][held], turned['left'][held][:, :, 0])
    angle = np.degrees(np.arctan2(-thumb[:, 2], thumb[:, 0]))
    out['left']['roll_deg'] = -float(np.median(angle))
    report['left_roll_std_deg'] = float(angle.std())
    turned['left'] = turned['left'] @ axis_rotation(Y, np.radians(out['left']['roll_deg']))
    # Constant error of the origins across the palm axis; each hand takes half.
    seen = np.einsum('nji,nj->ni', turned['right'][held], between)
    across = np.array([seen[:, 0].mean(), 0., seen[:, 2].mean()])
    out['left']['offset_m'] = (-across / 2).tolist(); out['right']['offset_m'] = (across / 2).tolist()
    report['origin_error_across_palm_axis_m'] = across.tolist()
    report['origin_error_std_m'] = seen.std(axis=0).tolist()
    report['held_frames'] = int(len(held)); report['rest_frames'] = int(len(rest))
    corrected = {side: poses[:, :3, 3] + np.einsum('nij,j->ni', turned[side], out[side]['offset_m'])
                 for side, poses in (('left', left), ('right', right))}
    height = corrected['left'][rest, 2] - corrected['right'][rest, 2]
    recorded = left[rest, 2, 3] - right[rest, 2, 3]
    report['left_minus_right_height_at_rest_m'] = dict(recorded=[float(recorded.mean()), float(recorded.std())],
                                                       corrected=[float(height.mean()), float(height.std())])
    return out, report


def main():
    from . import episode as episode_module, retarget, tracking
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='*', help='default: all')
    args = parser.parse_args()
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    episodes = []
    for index in (args.episodes if args.episodes else range(int(info['total_episodes']))):
        data, _ = tracking.clean(episode_module.load_episode(args.dataset, index))
        episodes.append((retarget.pose_matrices(data['left']), retarget.pose_matrices(data['right']),
                         episode_module.carry_segments(data)))
    constants, report = measure(episodes)
    print(json.dumps(dict(calibration=constants, report=report), indent=2))


if __name__ == '__main__':
    main()
