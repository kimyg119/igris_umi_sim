"""A pose-only stand-in for a policy trained on the labels: past hand poses in, a chunk of future hand poses out.

It sees no camera, so it is not the policy.  Its place in the test is the policy's: a network trained on the
preprocessed labels (labels.py) whose output - with the errors, the chunk boundaries and the delay a network has -
is what the controller is handed (stream.py).  The observation is the recent pose history of an episode the
network was not trained on, the way a trained policy is checked open loop: recorded observations in, predicted
actions out.  ``executed`` lays the predicted chunks out in time as a robot would run them: a new chunk every
``replan`` samples, each one arriving ``delay`` samples after the observation it was computed from.

  train  python -m igris.umi_walklow_replay.pose_policy train --labels <labels folder> --out <new folder>
  run    python -m igris.umi_walklow_replay.pose_policy run --labels <labels folder> --checkpoint <policy.pt> --out <new folder>

Rows are the dataset's state layout at the dataset rate (labels.state_rows): left and right hand xyz + xyzw in the
pelvis frame, then the trunk heading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .episode import continuous_quaternions
from .kinematics import axis_rotation, matrix_quat, quat_matrix, rotation_vector

HISTORY = 8                  # samples the network sees: 0.53 s at 15 Hz
HORIZON = 16                 # samples it predicts: 1.07 s
REPLAN = 4                   # a new chunk every 0.27 s
DELAY = 2                    # samples between an observation and the arrival of its chunk: 0.13 s
HELD_OUT = (7, 13, 21, 29)   # episodes kept out of training


def _rotations(row):
    return quat_matrix(row[[6, 3, 4, 5]]), quat_matrix(row[[13, 10, 11, 12]])


def observation(rows, latest):
    """Network input for the samples up to ``latest``: the latest pose (positions, the first two columns of each
    hand rotation, heading) and the ``HISTORY - 1`` before it with positions and heading relative to the latest.
    Before the start of an episode the first sample is repeated."""
    frames = []
    for index in range(latest - HISTORY + 1, latest + 1):
        row = rows[max(index, 0)]
        left, right = _rotations(row)
        frames.append(np.r_[row[0:3], left[:, :2].T.ravel(), row[7:10], right[:, :2].T.ravel(), row[14]])
    frames = np.array(frames)
    relative = frames[:-1].copy()
    relative[:, [0, 1, 2, 9, 10, 11, 18]] -= frames[-1, [0, 1, 2, 9, 10, 11, 18]]
    return np.r_[frames[-1], relative.ravel()].astype(np.float32)


def chunk_targets(rows, latest):
    """What the network is trained to give for ``latest``: for each of the next ``HORIZON`` samples the change of
    both hand positions, the turn of both hands (rotation vector in the latest hand frame) and the change of
    heading.  Past the end of the episode the last sample is repeated."""
    now = rows[latest]; base = _rotations(now)
    out = np.empty((HORIZON, 13))
    for step in range(HORIZON):
        row = rows[min(latest + 1 + step, len(rows) - 1)]
        turned = _rotations(row)
        out[step] = np.r_[row[0:3] - now[0:3], rotation_vector(base[0].T @ turned[0]),
                          row[7:10] - now[7:10], rotation_vector(base[1].T @ turned[1]), row[14] - now[14]]
    return out


def apply(now, chunk):
    """Rows (HORIZON x 15) that ``chunk`` describes from the pose ``now``."""
    base = _rotations(now)
    out = np.empty((len(chunk), 15))
    for step, values in enumerate(chunk):
        for hand in range(2):
            turn = values[6 * hand + 3:6 * hand + 6]; angle = np.linalg.norm(turn)
            rotation = base[hand] if angle < 1e-12 else base[hand] @ axis_rotation(turn / angle, angle)
            w, x, y, z = matrix_quat(rotation)
            out[step, 7 * hand:7 * hand + 3] = now[7 * hand:7 * hand + 3] + values[6 * hand:6 * hand + 3]
            out[step, 7 * hand + 3:7 * hand + 7] = [x, y, z, w]
        out[step, 14] = now[14] + values[12]
    return out


def _network(inputs, outputs, width):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(inputs, width), torch.nn.GELU(), torch.nn.Linear(width, width), torch.nn.GELU(),
                               torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, outputs))


def train(episodes, held_out=HELD_OUT, epochs=300, seed=0, width=512, batch=512, device='cpu'):
    """Train on ``episodes`` (episode index -> state rows) except ``held_out``.  Returns the checkpoint: weights,
    normalisation, settings and a report (the loss on the held-out episodes decides which epoch is kept)."""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    held_out = sorted(int(index) for index in held_out if int(index) in episodes)

    def pairs(indices):
        x = [observation(episodes[e], k) for e in indices for k in range(len(episodes[e]))]
        y = [chunk_targets(episodes[e], k).ravel() for e in indices for k in range(len(episodes[e]))]
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    x, y = pairs([index for index in sorted(episodes) if index not in held_out])
    x_val, y_val = pairs(held_out)
    mean_x, std_x = x.mean(0), x.std(0) + 1e-6
    scale_y = y.reshape(-1, HORIZON, 13).std(axis=(0, 1)) + 1e-6          # one scale per output kind, not per step
    scale_y = np.tile(scale_y, HORIZON).astype(np.float32)
    network = _network(x.shape[1], y.shape[1], width).to(device)
    tensors = [torch.tensor(v, device=device) for v in ((x - mean_x) / std_x, y / scale_y, (x_val - mean_x) / std_x, y_val / scale_y)]
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best = (np.inf, None, -1); first = None
    for epoch in range(epochs):
        network.train()
        order = torch.tensor(rng.permutation(len(x)), device=device)
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            loss = torch.nn.functional.smooth_l1_loss(network(tensors[0][rows]), tensors[1][rows])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        schedule.step()
        network.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.smooth_l1_loss(network(tensors[2]), tensors[3]))
        first = val if first is None else first
        if val < best[0]:
            best = (val, {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}, epoch)
    return dict(state=best[1], mean_x=mean_x, std_x=std_x, scale_y=scale_y, width=width, history=HISTORY, horizon=HORIZON,
                report=dict(held_out=held_out, train_windows=int(len(x)), val_windows=int(len(x_val)), epochs=epochs,
                            best_epoch=best[2], val_loss=best[0], first_val_loss=first, seed=seed))


def _loaded(saved):
    if '_network' not in saved:
        network = _network(len(saved['mean_x']), len(saved['scale_y']), saved['width'])
        network.load_state_dict(saved['state']); network.eval()
        saved['_network'] = network
    return saved['_network']


def predict(saved, rows, latest):
    """The chunk (HORIZON x 15 rows) the network gives for the samples up to ``latest``."""
    import torch
    with torch.no_grad():
        x = torch.tensor((observation(rows, latest) - saved['mean_x']) / saved['std_x']).float()[None]
        out = _loaded(saved)(x)[0].numpy() * saved['scale_y']
    return apply(rows[latest], out.reshape(HORIZON, 13))


def _mix(a, b, share):
    """Row between two state rows: positions and heading linear, hand rotations along the short way."""
    out = (1. - share) * a + share * b
    for columns in (slice(3, 7), slice(10, 14)):
        sign = 1. if np.dot(a[columns], b[columns]) >= 0. else -1.
        quaternion = (1. - share) * a[columns] + share * sign * b[columns]
        out[columns] = quaternion / np.linalg.norm(quaternion)
    return out


def executed(saved, rows, replan=REPLAN, delay=DELAY, details=False, blend=False):
    """The poses a controller is handed when the network's chunks are run on a robot (N x 15, one per sample).

    A chunk is computed from the samples up to k (every ``replan`` samples) and can be used from sample
    k + ``delay`` + 1 on; the newest usable chunk is always the one in use.  Until the first chunk has arrived
    the start pose is held.  ``details`` also returns, per sample, the index of the observation its pose came from.

    Two chunks disagree about the samples they both cover, and switching from one to the next makes the poses jump
    (up to 5-7 cm in one sample on held-out episodes).  With ``blend`` the new chunk is faded in over its own run
    against what the chunk before it said about the same samples.  That uses no observation the plain switch does
    not use, and adds no delay.
    """
    if delay + replan > HORIZON:
        raise ValueError('A chunk must last until the next one has arrived')
    out = np.repeat(rows[:1], len(rows), axis=0)
    source = np.zeros(len(rows), dtype=int)
    before = None                                                   # (chunk, the sample it was observed at)
    for observed in range(0, len(rows), replan):
        chunk = predict(saved, rows, observed)
        first = observed + delay + 1
        for offset, sample in enumerate(range(first, min(first + replan, len(rows)))):
            pose = chunk[sample - observed - 1]
            if blend and before is not None and sample - before[1] - 1 < HORIZON:
                pose = _mix(before[0][sample - before[1] - 1], pose, (offset + 1.) / replan)
            out[sample] = pose; source[sample] = observed
        before = (chunk, observed)
    out[:, 3:7] = continuous_quaternions(out[:, 3:7]); out[:, 10:14] = continuous_quaternions(out[:, 10:14])
    return (out, source) if details else out


def save(path, rows, rate_hz, **extra):
    """Write state rows as a stream file for ``replay --stream`` (labels.save_stream)."""
    from . import labels, retarget
    stream = dict(t=np.arange(len(rows)) / float(rate_hz), left=retarget.pose_matrices(rows[:, :7]),
                  right=retarget.pose_matrices(rows[:, 7:14]), head_yaw=rows[:, 14])
    if 'source_t' in extra:
        stream['source_t'] = extra.pop('source_t')
    labels.save_stream(path, stream, rate_hz, **extra)


def differences(stream, rows):
    """How far an executed stream is from the labels it was predicted for, and how large its jumps between
    chunks are (sample-to-sample change beyond that of the labels)."""
    position = np.linalg.norm(stream[:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3) - rows[:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3), axis=2)
    turn = np.array([[np.linalg.norm(rotation_vector(a.T @ b)) for a, b in zip(_rotations(x), _rotations(y))]
                     for x, y in zip(stream, rows)])
    step = np.linalg.norm(np.diff(stream[:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3), axis=0), axis=2)
    label_step = np.linalg.norm(np.diff(rows[:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3), axis=0), axis=2)
    return dict(position_p50_m=float(np.percentile(position, 50)), position_p99_m=float(np.percentile(position, 99)),
                position_max_m=float(position.max()), rotation_p99_deg=float(np.degrees(np.percentile(turn, 99))),
                step_p99_m=float(np.percentile(step, 99)), step_max_m=float(step.max()),
                label_step_p99_m=float(np.percentile(label_step, 99)), label_step_max_m=float(label_step.max()))


def load_labels(folder):
    """Episode index -> the label file's arrays, for every ``episode_XXX.npz`` of a labels folder."""
    return {int(path.stem.split('_')[1]): np.load(path) for path in sorted(Path(folder).glob('episode_*.npz'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['train', 'run'])
    parser.add_argument('--labels', type=Path, required=True, help='Folder written by labels.py')
    parser.add_argument('--out', type=Path, required=True, help='New folder')
    parser.add_argument('--checkpoint', type=Path, help='run: policy.pt written by train')
    parser.add_argument('--held-out', type=int, nargs='*', default=list(HELD_OUT))
    parser.add_argument('--episodes', type=int, nargs='*', help='run: episodes to predict (default: the held-out ones)')
    parser.add_argument('--signal', choices=['state', 'raw_state'], default='state',
                        help='train on / predict the labels (state) or the recording with constants only (raw_state)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replan', type=int, default=REPLAN)
    parser.add_argument('--delay', type=int, default=DELAY)
    parser.add_argument('--blend', action='store_true', help='run: fade each new chunk in against the one before it')
    args = parser.parse_args()
    import torch
    data = load_labels(args.labels)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.action == 'train':
        saved = train({index: file[args.signal] for index, file in data.items()}, args.held_out, args.epochs, args.seed)
        saved['report'].update(labels=str(args.labels), signal=args.signal)
        torch.save(saved, args.out / 'policy.pt')
        (args.out / 'report.json').write_text(json.dumps(saved['report'], indent=2) + '\n')
        print(json.dumps(saved['report'], indent=2))
        return
    saved = torch.load(args.checkpoint, weights_only=False)
    signal = saved['report'].get('signal', 'state')
    report = {}
    for index in args.episodes or saved['report']['held_out']:
        file = data[index]; rows = file[signal]
        stream = executed(saved, rows, args.replan, args.delay, blend=args.blend)
        rate = float(file['fps'])
        source = file['state_source_t'] if signal == 'state' else file['raw_t']
        save(args.out / f'episode_{index:03d}.npz', stream, rate, surface_top=float(file['surface_top']), source_t=source)
        report[index] = dict(differences(stream, rows), replan=args.replan, delay=args.delay, blend=args.blend, signal=signal,
                             trained_on=index not in saved['report']['held_out'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
