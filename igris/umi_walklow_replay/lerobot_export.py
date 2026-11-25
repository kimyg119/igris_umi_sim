"""The simulated replays as one LeRobot dataset: the EE dataset made through the robot's links, with its cameras.

Every successful run that ``ee_dataset.py`` has made numbers for becomes an episode at 15 Hz.  The pictures are
drawn again at full size from the recorded body poses (the run folder keeps them only as compressed video).

  observation.images.head_cam / left_cam / right_cam    480 x 640, the cameras of cameras.py (names of the UMI dataset)
  observation.state      30  hands in the chest frame (position + two rotation columns each), the right hand in the
                             left hand's frame, waist joint angles
  observation.ee_pelvis  18  the same two hands in the pelvis frame: what relative actions are composed with on a robot
  observation.ee_room    18  ... and in the room (simulator pelvis pose): what the relative labels are made with
  observation.joints     31  measured joint angles, SDK order
  action                 21  what the controller followed: hand targets in the pelvis frame, waist joint targets
  action.room            18  the hand targets in the room
  action.relative        21  the one-step relative action of ee_features.py: the left hand's target from the left hand of this
                             frame, the right target in the left target's frame, the waist target minus the waist now
  action.joints          31  commanded joint angles
  source.episode          1  episode of the UMI dataset the run replays

A chunk of relative actions is always relative to the frame it starts at, so it is not a column: it is made
from ``observation.ee_room`` (or ``ee_pelvis``) of that frame and ``action.room`` (or ``action``) of the frames after
it with ``ee_features.relative_actions`` - ``act_ee.py`` does exactly that.

  python -m igris.umi_walklow_replay.lerobot_export --runs <folder of run outputs> --out <new dataset folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import cameras, ee_features

TASK = 'Move the three parcels from the case onto the roll container.'
HANDS = [f'{hand}_ee.{part}' for hand in ('left', 'right') for part in ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21')]


def features(height=480, width=640):
    from .kinematics import JOINTS
    video = dict(dtype='video', shape=(height, width, 3), names=['height', 'width', 'channels'])
    waist = ['waist_yaw', 'waist_roll', 'waist_pitch']
    return {
        **{f'observation.images.{name}': dict(video) for name in cameras.NAMES},
        'observation.state': dict(dtype='float32', shape=(ee_features.STATE_SIZE,), names=list(ee_features.STATE_NAMES)),
        'observation.ee_pelvis': dict(dtype='float32', shape=(18,), names=HANDS),
        'observation.ee_room': dict(dtype='float32', shape=(18,), names=HANDS),
        'observation.joints': dict(dtype='float32', shape=(31,), names=list(JOINTS)),
        'action': dict(dtype='float32', shape=(21,), names=HANDS + waist),
        'action.room': dict(dtype='float32', shape=(18,), names=HANDS),
        'action.relative': dict(dtype='float32', shape=(21,), names=list(ee_features.ACTION_NAMES)),
        'action.joints': dict(dtype='float32', shape=(31,), names=list(JOINTS)),
        'source.episode': dict(dtype='int64', shape=(1,), names=None),
    }


def rows(data):
    """Per-frame feature arrays (without the pictures) of one run's ``frames.npz``."""
    hands9 = lambda poses: np.concatenate([ee_features.pose9(poses[:, 0]), ee_features.pose9(poses[:, 1])], axis=1)      # noqa: E731
    waist_now = data['q_measured'][:, ee_features.WAIST]
    relative = np.array([ee_features.relative_actions(data['ee_room'][k], waist_now[k], data['target_room'][k:k + 1],
                                                      data['waist_target'][k:k + 1])[0] for k in range(len(waist_now))])
    return {
        'observation.state': ee_features.state(data['ee_pelvis'], data['chest_pelvis'], waist_now).astype(np.float32),
        'observation.ee_pelvis': hands9(data['ee_pelvis']).astype(np.float32),
        'observation.ee_room': hands9(data['ee_room']).astype(np.float32),
        'observation.joints': data['q_measured'].astype(np.float32),
        'action': np.concatenate([hands9(data['target_pelvis']), data['waist_target']], axis=1).astype(np.float32),
        'action.room': hands9(data['target_room']).astype(np.float32),
        'action.relative': relative.astype(np.float32),
        'action.joints': data['q_command'].astype(np.float32),
        'source.episode': np.full((len(waist_now), 1), int(data['episode']), dtype=np.int64),
    }


def export(runs, out, repo_id='igris_c/umi_box_palletizing_sim_ee', limit=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .act_ee import find_runs
    found = find_runs(runs)[:limit]
    if not found:
        raise ValueError('No successful rendered runs found')
    dataset = LeRobotDataset.create(repo_id, fps=int(round(found[0][1]['rate_hz'])), features=features(), root=Path(out),
                                    robot_type='igris_c_sim', use_videos=True, streaming_encoding=True, encoder_threads=4)
    written = []
    for path, info in found:
        data = np.load(path / 'dataset/frames.npz')
        columns = rows(data)
        _, _, poses = cameras.read_poses(path / 'run')
        view = cameras.Cameras(path / 'run')
        try:
            for index, record in enumerate(data['camera_record']):
                frame = {key: value[index] for key, value in columns.items()}
                for name, picture in view.render(poses[record]).items():
                    frame[f'observation.images.{name}'] = picture
                frame['task'] = TASK
                dataset.add_frame(frame)
        finally:
            view.close()
        dataset.save_episode()
        written.append(dict(run=path.name, source_episode=int(data['episode']), frames=int(len(data['t']))))
        print(json.dumps(written[-1]), flush=True)
    dataset.finalize()
    (Path(out) / 'meta/sim_runs.json').write_text(json.dumps(dict(task=TASK, runs=written), indent=2) + '\n')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New dataset folder')
    parser.add_argument('--repo-id', default='igris_c/umi_box_palletizing_sim_ee')
    parser.add_argument('--limit', type=int, help='Only the first N runs')
    args = parser.parse_args()
    written = export(args.runs, args.out, args.repo_id, args.limit)
    print(json.dumps(dict(episodes=len(written), frames=sum(row['frames'] for row in written))))


if __name__ == '__main__':
    main()
