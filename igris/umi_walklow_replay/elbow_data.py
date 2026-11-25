"""FiSTA training windows from the teleoperated joint recordings of this robot.

The elbow prior of ``igris/hl_ik`` (FiSTA) was trained on retargeted human motion, and the UMI box episodes sit
at the edge of what it saw.  This module packs the robot's own teleoperated box handling
(``kimyg119/Igris_Teleop_Box_Palletizing``, tag ``full-signals-15hz``: 834 episodes, 31 commanded joints) in the
layout ``igris.hl_ik.train`` reads, so the same network can be trained on the arm postures the operators
actually used for this task:

  packed/<scope>/{train,val,test}/states.npy   frames x 2 arms x 14: hand pose, elbow pose (xyz + wxyz),
                                               torso axes, measured from the shoulder, 30 Hz
  packed/<scope>/{train,val,test}/starts.npy   first frame of every window of six frames inside one episode
  packed/<scope>/{train,val,test}/clips.json   the episodes in the order they are stored
  packed/<scope>/manifest.json

The recording is at 15 Hz and the network steps at 30 Hz, so the joint commands are doubled in rate
(Catmull-Rom midpoints) before the poses are taken.  The poses are exactly ``ElbowPrior.state``, i.e. what the
prior is fed at run time.  Train, validation and test never share a recording session.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .elbow_prior import ElbowPrior
from .kinematics import JOINTS, UpperBody
from .teleop import sdk_name

WINDOW = 6                   # five history frames and the frame to predict
RATE_HZ = 30.                # step of the network


def double_rate(q):
    """Samples at twice the rate (2N - 1 rows): the originals with Catmull-Rom midpoints in between."""
    q = np.asarray(q, dtype=float)
    padded = np.concatenate([2 * q[:1] - q[1:2], q, 2 * q[-1:] - q[-2:-1]])       # straight continuation at both ends
    middle = (-padded[:-3] + 9 * padded[1:-2] + 9 * padded[2:-1] - padded[3:]) / 16.
    out = np.empty((2 * len(q) - 1, q.shape[1]))
    out[::2] = q; out[1::2] = middle
    return out


def states(model, q):
    """Network states (N x 2 x 14, float32) of 31-joint configurations: ``ElbowPrior.state`` row by row."""
    prior = ElbowPrior(model, None)
    return np.array([prior.state(row) for row in np.asarray(q, dtype=float)], dtype=np.float32)


def window_starts(lengths):
    """First frames of all windows that lie inside one episode, for episodes stored end to end."""
    starts = []; cursor = 0
    for length in lengths:
        starts.extend(range(cursor, cursor + max(0, length - WINDOW + 1)))
        cursor += length
    return np.array(starts, dtype=np.int64)


def split_sessions(frames, val=.1, test=.1, seed=0):
    """Session -> 'train' | 'val' | 'test'.  Whole sessions, in an order fixed by a hash of the session number:
    the first ones fill the test share of the frames, the next ones the validation share."""
    order = sorted(frames, key=lambda session: hashlib.sha256(f'{seed}:{session}'.encode()).hexdigest())
    total = float(sum(frames.values()))
    out = {}; taken = {'test': 0., 'val': 0.}
    for session in order:
        where = 'test' if taken['test'] < test * total else 'val' if taken['val'] < val * total else 'train'
        if where != 'train':
            taken[where] += frames[session]
        out[session] = where
    return out


_model = None


def _episode_states(action):
    global _model
    if _model is None:
        _model = UpperBody.from_mjcf()
    return states(_model, double_rate(action))


def pack(model, episodes, split, out, jobs=1):
    """Write the packed splits for ``episodes`` (dicts with ``episode``, ``session``, ``action`` N x 31 at the
    recorded rate) and return the per-split summary.  ``split`` maps a session to its split."""
    out = Path(out)
    episodes = sorted(episodes, key=lambda item: item['episode'])
    if jobs > 1:
        with Pool(jobs) as pool:
            computed = pool.map(_episode_states, [item['action'] for item in episodes], chunksize=4)
    else:
        computed = [states(model, double_rate(item['action'])) for item in episodes]
    summary = {}
    for name in ('train', 'val', 'test'):
        chosen = [(item, state) for item, state in zip(episodes, computed) if split[item['session']] == name]
        if not chosen:
            raise ValueError(f'No episode in the {name} split')
        folder = out / name; folder.mkdir(parents=True, exist_ok=False)
        lengths = [len(state) for _, state in chosen]
        np.save(folder / 'states.npy', np.concatenate([state for _, state in chosen]))
        starts = window_starts(lengths)
        np.save(folder / 'starts.npy', starts)
        clips = []; cursor = 0
        for (item, _), length in zip(chosen, lengths):
            clips.append(dict(episode=int(item['episode']), session=int(item['session']), start=cursor, frames=length,
                              windows=max(0, length - WINDOW + 1)))
            cursor += length
        (folder / 'clips.json').write_text(json.dumps(clips, indent=1) + '\n')
        summary[name] = dict(episodes=len(chosen), frames=int(sum(lengths)), windows=int(len(starts)),
                             sessions=sorted({int(item['session']) for item, _ in chosen}))
    manifest = dict(summary, rate_hz=RATE_HZ, window=WINDOW, signal='commanded joints (action), rate doubled with Catmull-Rom midpoints',
                    state='ElbowPrior.state: hand and elbow pose per arm, torso axes, from the shoulder',
                    split='whole recording sessions; test and validation each about a tenth of the frames')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return summary


def load_teleop(root):
    """Every episode of the teleop dataset: dicts with ``episode``, ``session``, ``task``, ``action`` (N x 31)."""
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    if [sdk_name(name) for name in info['features']['action']['names']] != JOINTS:
        raise ValueError('action is not the 31-joint SDK order; use the full-signals-15hz revision')
    rows = {row['episode_index']: row for path in sorted((root / 'meta/episodes').rglob('*.parquet'))
            for row in pq.read_table(path, columns=['episode_index', 'tasks', 'length', 'source/session_index']).to_pylist()}
    episodes = []
    for path in sorted((root / 'data').rglob('*.parquet')):
        table = pq.read_table(path, columns=['episode_index', 'action'])
        index = table.column('episode_index').to_numpy()
        action = np.array(table.column('action').to_pylist(), dtype=float)
        for episode in np.unique(index):
            row = rows[int(episode)]
            chosen = action[index == episode]
            if len(chosen) != row['length']:
                raise ValueError(f'Episode {episode}: {len(chosen)} rows found, {row["length"]} expected')
            episodes.append(dict(episode=int(episode), session=int(row['source/session_index']),
                                 task=row['tasks'][0] if row['tasks'] else '', action=chosen))
    return float(info['fps']), episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True, help='Local copy of the teleop dataset (full-signals-15hz)')
    parser.add_argument('--out', type=Path, required=True, help='Experiment root; packed/<scope>/ is written below it')
    parser.add_argument('--scope', default='teleop')
    parser.add_argument('--jobs', type=int, default=16)
    args = parser.parse_args()
    fps, episodes = load_teleop(args.dataset)
    if abs(2 * fps - RATE_HZ) > 1e-9:
        raise ValueError(f'The recording is at {fps} Hz; this packer doubles it to reach {RATE_HZ} Hz')
    episodes = [item for item in episodes if 2 * len(item['action']) - 1 >= WINDOW]
    frames = {}
    for item in episodes:
        frames[item['session']] = frames.get(item['session'], 0) + len(item['action'])
    split = split_sessions(frames)
    target = args.out / 'packed' / args.scope
    summary = pack(UpperBody.from_mjcf(), episodes, split, target, jobs=args.jobs)
    manifest = json.loads((target / 'manifest.json').read_text())
    manifest.update(dataset=str(args.dataset), recorded_rate_hz=fps, episodes_total=len(episodes),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
