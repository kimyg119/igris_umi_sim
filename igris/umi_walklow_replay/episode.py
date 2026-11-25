"""Extract one UMI episode and the pick/place geometry it implies.

The dataset stores, per frame, the left/right hand pose (xyz + xyzw quaternion)
in the robot pelvis frame plus the operator's head yaw.  No joint values exist.
Run with a Python that has pyarrow (``env_isaaclab/bin/python``).
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATASET = 'rcvsungmincho/Box_palletizing_UMI_macos_intern_merged'
# Summed 8x32 pressure on both palms decides "holding the box".  A grasp starts
# when the weaker palm exceeds TACTILE_ON and ends when it falls below
# TACTILE_OFF.  Free motion leaves 0.4-2.5 on the weaker palm; a light grasp
# (episode 2, second box) hovers at 5.5-12, so one threshold would split it.
TACTILE_ON = 8.0
TACTILE_OFF = 3.5
MIN_CARRY_S = 3.0
# A held box keeps the palms at a fixed distance.  The pressure of a light grasp
# can rise seconds after the box has left its support (episodes 10 and 20), so
# each tactile interval is extended over the neighbouring frames whose palm
# distance stays within this band of the interval's median.
HOLD_BAND_M = 0.015


def continuous_quaternions(q):
    """Flip signs so neighbouring samples stay on the same hemisphere."""
    q = np.array(q, dtype=float, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def load_episode(root, episode):
    import pyarrow.dataset as ds
    root = Path(root)
    if not (root / 'data').is_dir():
        raise FileNotFoundError(f'No LeRobot data directory under {root}')
    info = json.loads((root / 'meta/info.json').read_text())
    if not 0 <= episode < info['total_episodes']:
        raise ValueError(f"Episode {episode} is outside 0..{info['total_episodes'] - 1}")
    names = info['features']['observation.state']['names']
    expected = [f'{s}_ee.{c}' for s in ('left', 'right') for c in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] + ['head.yaw']
    if names != expected:
        raise ValueError(f'Unexpected state layout: {names}')
    columns = ['observation.state', 'observation.tactile_left_raw', 'observation.tactile_right_raw',
               'timestamp', 'frame_index']
    table = ds.dataset(root / 'data', format='parquet').to_table(
        columns=columns, filter=ds.field('episode_index') == episode)
    order = np.argsort(table['frame_index'].to_numpy())
    column = lambda name: np.stack(table[name].to_numpy(zero_copy_only=False))[order]
    state = column('observation.state').astype(float)
    if len(state) < 2 or not np.isfinite(state).all():
        raise ValueError('Episode is empty or contains non-finite poses')
    frames = table['frame_index'].to_numpy()[order]
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError('Episode frames are not contiguous')
    left, right = state[:, :7].copy(), state[:, 7:14].copy()
    for pose in (left, right):
        norm = np.linalg.norm(pose[:, 3:], axis=1)
        if np.any(abs(norm - 1) > 1e-3):
            raise ValueError('Non-unit hand quaternion')
        pose[:, 3:] = continuous_quaternions(pose[:, 3:] / norm[:, None])
    return dict(t=np.arange(len(state)) / float(info['fps']), fps=float(info['fps']), left=left, right=right,
                head_yaw=np.unwrap(state[:, 14]),
                tactile_left=column('observation.tactile_left_raw').sum(1).astype(float),
                tactile_right=column('observation.tactile_right_raw').sum(1).astype(float))


def carry_segments(episode, threshold=TACTILE_ON, release=TACTILE_OFF, minimum_s=MIN_CARRY_S):
    """Intervals in which both palms press the box, with their end geometry."""
    fps = episode['fps']
    weaker = np.minimum(episode['tactile_left'], episode['tactile_right'])
    held = np.zeros(len(weaker), dtype=bool)
    holding = False
    for i, value in enumerate(weaker):
        holding = value > release if holding else value > threshold
        held[i] = holding
    # Bridge dropouts shorter than a third of a second.
    gap = max(1, int(round(fps / 3)))
    edges = np.flatnonzero(np.diff(np.r_[0, held.astype(int), 0]))
    spans = [[a, b] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    left, right = episode['left'][:, :3], episode['right'][:, :3]
    middle = (left + right) / 2
    distance = np.linalg.norm(left - right, axis=1)
    merged = [span for span in merged if (span[1] - span[0]) / fps >= minimum_s]
    for k, span in enumerate(merged):
        limit = np.median(distance[span[0]:span[1]]) + HOLD_BAND_M
        floor = merged[k - 1][1] if k else 0
        while span[0] > floor and distance[span[0] - 1] < limit:
            span[0] -= 1
        # Pressure can linger while the hands already open: within the last second,
        # end where the palms part.  (Mid-carry regrips also widen the palms.)
        tail = max(span[0], span[1] - int(round(fps)))
        parted = np.flatnonzero(distance[tail:span[1]] >= limit)
        if len(parted):
            span[1] = tail + int(parted[0])
    segments = []
    for a, b in merged:
        if (b - a) / fps < minimum_s:
            continue
        window = max(2, int(round(fps / 3)))
        first, last = slice(a, a + window), slice(b - window, b)
        axis = lambda s: (left[s] - right[s]).mean(0) / np.linalg.norm((left[s] - right[s]).mean(0))
        segments.append(dict(pick_index=int(a), place_index=int(b - 1),
                             pick_mid=middle[first].mean(0), place_mid=middle[last].mean(0),
                             axis_pick=axis(first), axis_place=axis(last),
                             separation=float(np.median(np.linalg.norm(left[a:b] - right[a:b], axis=1)))))
    return segments


def geometry(episode):
    """Scene quantities in the pelvis frame, derived only from this episode."""
    segments = carry_segments(episode)
    if not segments:
        raise ValueError('No bimanual carry was found in this episode')
    pick = np.array([s['pick_mid'] for s in segments]); place = np.array([s['place_mid'] for s in segments])
    report = dict(boxes=len(segments), separation_m=float(np.median([s['separation'] for s in segments])),
                  pick_xy=pick[:, :2].mean(0).tolist(), place_xy=place[:, :2].mean(0).tolist(),
                  pick_z=pick[:, 2].tolist(), place_z=place[:, 2].tolist(),
                  pick_axis_yaw_rad=float(np.mean([np.arctan2(s['axis_pick'][1], s['axis_pick'][0]) for s in segments])),
                  place_axis_yaw_rad=float(np.mean([np.arctan2(s['axis_place'][1], s['axis_place'][0]) for s in segments])),
                  pick_axis_yaws_rad=[float(np.arctan2(s['axis_pick'][1], s['axis_pick'][0])) for s in segments],
                  place_axis_yaws_rad=[float(np.arctan2(s['axis_place'][1], s['axis_place'][0])) for s in segments],
                  carry_frames=[[s['pick_index'], s['place_index']] for s in segments])
    if len(segments) >= 3:
        # Box k is taken from level n-1-k and set down on level k.  With a rigid
        # grasp the grip height cancels: pick_k - place_k = dz + (n-1-2k) * height.
        n = len(segments); levels = n - 1 - 2 * np.arange(n)
        height, offset = np.linalg.lstsq(np.c_[levels, np.ones(n)], pick[:, 2] - place[:, 2], rcond=None)[0]
        report.update(box_height_m=float(height), source_minus_destination_top_m=float(offset))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Local LeRobot dataset directory')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'episode.npz'
    if target.exists():
        raise FileExistsError(target)
    episode = load_episode(args.root, args.episode)
    report = dict(dataset=DATASET, root=str(args.root), episode=args.episode, frames=len(episode['t']),
                  duration_s=float(episode['t'][-1]), **geometry(episode))
    np.savez_compressed(target, **{k: v for k, v in episode.items() if k != 'fps'}, fps=episode['fps'])
    (args.out / 'episode.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
