"""ACT on relative end-effector actions, trained on the simulated replays.

Input: the three robot cameras (cameras.py) and the 21 state values of ee_features.py.  Output: a chunk of
relative actions - both hands in their own present frame, the waist joints as a change from now.  The network
is LeRobot's ACT (``lerobot.policies.act``); this module is the data side of it, which LeRobot's own pipeline
does not have: an action chunk that is relative to the pose at the moment of its observation cannot be stored
per frame, it has to be made per (frame, chunk).

  python -m igris.umi_walklow_replay.act_ee train --runs <folder of run outputs> --out <new folder> [--held-out 7 13 21 29]
  python -m igris.umi_walklow_replay.act_ee check --checkpoint <folder> --runs <folder>      open-loop error on the held-out runs

``--frame room`` (default) makes the labels from the hands and their targets in the room, ``--frame pelvis`` from
both in the pelvis frame as commanded (no simulator pose in the labels; the pelvis motion within a chunk is then
part of what the network has to guess).  ``--absolute`` trains the same network on pelvis-frame targets as they
are, for comparison, and ``--independent-hands`` gives each hand its own relative action instead of the left hand
and the pair.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import ee_dataset, ee_features

CHUNK = 32                         # 2.1 s at 15 Hz
HELD_OUT = (7, 13, 21, 29)
CAMERAS = ('head_cam', 'left_cam', 'right_cam')
IMAGE_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGE_STD = np.array([.229, .224, .225], dtype=np.float32)
STATE_NOISE = .03                  # in standard deviations of each state value, training only
SHIFT_PIXELS = 8                   # largest random shift of a training picture


def chunks(now, waist_now, targets, waist_targets, length=CHUNK, paired=True):
    """Relative action chunks of one episode: (T x length x 21, T x length pad mask).  ``now`` and ``targets`` are
    T x 2 x 4 x 4 in one common frame; past the end of the episode the last target is repeated and marked.
    The same numbers as ``ee_features.relative_actions`` gives frame by frame."""
    count = len(now)
    index = np.minimum(np.arange(count)[:, None] + np.arange(length)[None], count - 1)          # T x length
    pad = np.arange(count)[:, None] + np.arange(length)[None] > count - 1
    picked = targets[index]                                                                      # T x length x 2 x 4 x 4
    left = ee_features.inverse(now[:, 0])[:, None] @ picked[:, :, 0]
    right = ee_features.inverse(picked[:, :, 0]) @ picked[:, :, 1] if paired else ee_features.inverse(now[:, 1])[:, None] @ picked[:, :, 1]
    out = np.concatenate([ee_features.pose9(left), ee_features.pose9(right), waist_targets[index] - waist_now[:, None]], axis=-1)
    return out.astype(np.float32), pad


def absolute_chunks(targets, waist_targets, length=CHUNK, paired=True):
    """The same chunks without the reference: the left hand's target and the waist targets as they are (for the
    comparison with absolute actions).  The right hand is still given in the left target's frame when ``paired``."""
    count = len(targets)
    index = np.minimum(np.arange(count)[:, None] + np.arange(length)[None], count - 1)
    pad = np.arange(count)[:, None] + np.arange(length)[None] > count - 1
    picked = targets[index]
    right = ee_features.inverse(picked[:, :, 0]) @ picked[:, :, 1] if paired else picked[:, :, 1]
    out = np.concatenate([ee_features.pose9(picked[:, :, 0]), ee_features.pose9(right), waist_targets[index]], axis=-1)
    return out.astype(np.float32), pad


def find_runs(folder, only_success=True):
    """Run outputs under ``folder`` that have a rendered dataset: list of (path, summary)."""
    found = []
    for summary in sorted(Path(folder).glob('*/dataset/images.json')):
        info = json.loads(summary.read_text())
        if info.get('stack_success') or not only_success:
            found.append((summary.parent.parent, info))
    return found


class Episode:
    """One rendered run: numbers in memory, pictures read from the JPEG stores on demand."""

    def __init__(self, path, info, frame='room', absolute=False, length=CHUNK, paired=True):
        self.path = Path(path); self.info = info
        data = np.load(self.path / 'dataset/frames.npz')
        self.episode = int(data['episode']); self.count = len(data['t'])
        waist_now = data['q_measured'][:, ee_features.WAIST]
        # From the saved link poses, not the saved ``state``: the definition of the state has grown since some were rendered.
        self.state = ee_features.state(data['ee_pelvis'], data['chest_pelvis'], waist_now).astype(np.float32)
        if absolute:
            self.actions, self.pad = absolute_chunks(data['target_pelvis'], data['waist_target'], length, paired)
        else:
            self.actions, self.pad = chunks(data[f'ee_{frame}'], waist_now, data[f'target_{frame}'], data['waist_target'], length, paired)
        self.offsets = {name: np.asarray(info['offsets'][name]) for name in CAMERAS}
        self.blobs = {}

    def picture(self, name, index):
        if name not in self.blobs:                              # opened in the process that reads (data loader workers)
            self.blobs[name] = np.memmap(self.path / f'dataset/{name}.jpgs', dtype=np.uint8, mode='r')
        return ee_dataset.read_jpeg(self.blobs[name], self.offsets[name], index)


def statistics(episodes):
    """Mean and standard deviation of the state values and of the action values (over every step of every
    chunk that is not padding), per value."""
    state = np.concatenate([episode.state for episode in episodes])
    actions = np.concatenate([episode.actions[~episode.pad] for episode in episodes])
    return dict(state_mean=state.mean(0), state_std=np.maximum(state.std(0), 1e-4),
                action_mean=actions.mean(0), action_std=np.maximum(actions.std(0), 1e-4))


def vary_picture(picture, shift=None, gain=None):
    """A training picture moved by ``shift`` (dx, dy; edge pixels repeated) and changed by ``gain`` (contrast,
    brightness as a share of full scale); uint8 H x W x 3 in and out."""
    image = picture
    if shift is not None and (shift[0] or shift[1]):
        height, width = image.shape[:2]
        rows = np.clip(np.arange(height) - shift[1], 0, height - 1); columns = np.clip(np.arange(width) - shift[0], 0, width - 1)
        image = image[rows][:, columns]
    if gain is not None:
        image = np.clip((image.astype(np.float32) - 127.5) * gain[0] + 127.5 + 255. * gain[1], 0., 255.).astype(np.uint8)
    return np.ascontiguousarray(image)


def normalise(pictures):
    """uint8 pictures (B x H x W x 3, a tensor on the device of the network) -> float B x 3 x H x W, ImageNet-normalised.
    Done on the GPU: as floats a batch of three cameras is 166 MB to hand from a loader process to the trainer."""
    import torch
    mean = torch.as_tensor(IMAGE_MEAN, device=pictures.device); std = torch.as_tensor(IMAGE_STD, device=pictures.device)
    return ((pictures.float() / 255. - mean) / std).permute(0, 3, 1, 2).contiguous()


class Windows:
    """Training samples: every frame of every episode (a ``torch.utils.data.Dataset`` by duck typing)."""

    def __init__(self, episodes, stats, augment=False, seed=0):
        self.episodes, self.stats, self.augment = episodes, stats, augment
        self.index = np.array([(e, t) for e, episode in enumerate(episodes) for t in range(episode.count)], dtype=np.int64)
        self.seed = seed

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        e, t = self.index[item]
        episode = self.episodes[e]
        rng = np.random.default_rng((self.seed, int(item), int(time.time_ns() % (1 << 31)))) if self.augment else None
        sample = {}
        for name in CAMERAS:
            shift = rng.integers(-SHIFT_PIXELS, SHIFT_PIXELS + 1, size=2) if self.augment else None
            gain = (rng.uniform(.85, 1.15), rng.uniform(-.08, .08)) if self.augment else None
            sample[f'observation.images.{name}'] = vary_picture(episode.picture(name, t), shift, gain)
        state = (episode.state[t] - self.stats['state_mean']) / self.stats['state_std']
        if self.augment:
            state = state + rng.normal(size=state.shape) * STATE_NOISE
        sample['observation.state'] = state.astype(np.float32)
        sample['action'] = ((episode.actions[t] - self.stats['action_mean']) / self.stats['action_std']).astype(np.float32)
        sample['action_is_pad'] = episode.pad[t]
        return sample


def build(height, width, chunk=CHUNK, device='cuda', **overrides):
    """A LeRobot ACT policy for the three cameras, the 21 state values and chunks of 21 action values."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    inputs = {'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(ee_features.STATE_SIZE,))}
    for name in CAMERAS:
        inputs[f'observation.images.{name}'] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
    config = ACTConfig(input_features=inputs, output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(ee_features.ACTION_SIZE,))},
                       chunk_size=chunk, n_action_steps=chunk, device=device, **overrides)
    return ACTPolicy(config)


def errors(predicted, wanted, pad, steps=slice(2, 10)):
    """Physical size of the prediction error over the chunk steps a controller executes (2..9 by default): position
    [m] and rotation [deg] of the two pose blocks of an action (the left hand; the second block is the pair, or the
    right hand without pairing), and the waist [deg].  Means over steps and samples that are not padding."""
    keep = ~pad[:, steps]
    report = {}
    for block, name in enumerate(('first', 'second')):
        a = ee_features.from9(predicted[:, steps, 9 * block:9 * block + 9]); b = ee_features.from9(wanted[:, steps, 9 * block:9 * block + 9])
        position = np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1)[keep]
        cosine = (np.einsum('...ij,...ij->...', a[..., :3, :3], b[..., :3, :3]) - 1.) / 2.
        turn = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))[keep]
        report.update({f'{name}_position_mean_m': float(position.mean()), f'{name}_position_p95_m': float(np.percentile(position, 95)),
                       f'{name}_rotation_mean_deg': float(turn.mean()), f'{name}_rotation_p95_deg': float(np.percentile(turn, 95))})
    waist = np.degrees(np.abs(predicted[:, steps, 18:] - wanted[:, steps, 18:]))[keep]
    report.update(waist_mean_deg=float(waist.mean()), waist_p95_deg=float(np.percentile(waist, 95)))
    return report


def quiet_worker(_):
    """A data loader worker decodes pictures on one thread: OpenCV would start one per CPU of the machine in each."""
    import cv2
    cv2.setNumThreads(0)


def to_device(batch, device):
    """A batch of ``Windows`` samples on the device of the network, its pictures as the network takes them."""
    import torch
    out = {key: (value if torch.is_tensor(value) else torch.as_tensor(value)).to(device, non_blocking=True) for key, value in batch.items()}
    for key in list(out):
        if key.startswith('observation.images.'):
            out[key] = normalise(out[key])
    return out


def evaluate(policy, loader, stats, device, limit=None):
    """Loss and physical errors over a loader of held-out samples."""
    import torch
    policy.eval()
    predicted = []; wanted = []; pads = []; losses = []
    with torch.no_grad():
        for count, batch in enumerate(loader):
            if limit and count >= limit:
                break
            batch = to_device(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                out = policy.predict_action_chunk(batch)
            out = out.float()
            valid = ~batch['action_is_pad'].unsqueeze(-1)
            losses.append(float(((out - batch['action']).abs() * valid).sum() / (valid.sum() * out.shape[-1]).clamp_min(1)))
            predicted.append(out.cpu().numpy() * stats['action_std'] + stats['action_mean'])
            wanted.append(batch['action'].cpu().numpy() * stats['action_std'] + stats['action_mean'])
            pads.append(batch['action_is_pad'].cpu().numpy())
    policy.train()
    report = errors(np.concatenate(predicted), np.concatenate(wanted), np.concatenate(pads)) if predicted else {}
    return dict(l1=float(np.mean(losses)) if losses else None, **report)


def save(policy, stats, folder, **meta):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(folder)
    np.savez(folder / 'stats.npz', **stats)
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def train(runs, out, held_out=HELD_OUT, steps=30000, batch_size=64, lr=5e-5, frame='room', absolute=False, workers=12,
          seed=0, device='cuda', check_every=1000, chunk=CHUNK, limit_runs=None, paired=True):
    import torch
    from torch.utils.data import DataLoader
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); np.random.seed(seed)
    found = find_runs(runs)
    if limit_runs:
        found = found[:limit_runs]
    episodes = [Episode(path, info, frame, absolute, chunk, paired) for path, info in found]
    training = [episode for episode in episodes if episode.episode not in held_out]
    held = [episode for episode in episodes if episode.episode in held_out]
    if not training:
        raise ValueError('No training runs found')
    stats = statistics(training)
    width, height = found[0][1]['small']
    torch.set_num_threads(4)
    loader = DataLoader(Windows(training, stats, augment=True, seed=seed), batch_size=batch_size, shuffle=True, num_workers=workers,
                        drop_last=True, persistent_workers=workers > 0, pin_memory=True, prefetch_factor=2 if workers else None,
                        worker_init_fn=quiet_worker if workers else None)
    # The check reads the same 40 batches every time, spread over the held-out runs.
    held_set = Windows(held, stats) if held else None
    if held_set is not None and len(held_set) > 40 * batch_size:
        held_set.index = held_set.index[np.linspace(0, len(held_set) - 1, 40 * batch_size).astype(int)]
    held_loader = DataLoader(held_set, batch_size=batch_size, shuffle=False, num_workers=min(workers, 4),
                             persistent_workers=workers > 0, worker_init_fn=quiet_worker if workers else None) if held else None
    policy = build(height, width, chunk, device, optimizer_lr=lr, optimizer_lr_backbone=lr).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=lr, weight_decay=policy.config.optimizer_weight_decay)
    warm = 500
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / warm) * .5 * (1 + np.cos(np.pi * min(1., step / steps))))
    meta = dict(runs=str(runs), held_out=list(held_out), training_runs=len(training), held_out_runs=len(held),
                training_frames=int(sum(e.count for e in training)), held_out_frames=int(sum(e.count for e in held)),
                frame=frame, absolute=bool(absolute), paired=bool(paired), chunk=chunk, rate_hz=float(found[0][1]['rate_hz']), small=[width, height],
                steps=steps, batch_size=batch_size, lr=lr, seed=seed, cameras=list(CAMERAS),
                state_names=ee_features.STATE_NAMES, action_names=ee_features.ACTION_NAMES)
    print('TRAIN', json.dumps({key: meta[key] for key in ('training_runs', 'held_out_runs', 'training_frames', 'held_out_frames', 'frame', 'absolute', 'paired')}), flush=True)
    history = []; best = None; step = 0; started = time.monotonic(); running = []
    while step < steps:
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                loss, parts = policy.forward(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
            optimizer.step(); schedule.step(); step += 1
            running.append(parts['l1_loss'])
            if step % check_every == 0 or step == steps:
                row = dict(step=step, train_l1=float(np.mean(running)), kld=parts.get('kld_loss'), minutes=(time.monotonic() - started) / 60.)
                running = []
                clock = time.monotonic()
                if held_loader is not None:
                    row.update({f'held_{key}': value for key, value in evaluate(policy, held_loader, stats, device, limit=40).items()})
                row['check_seconds'] = time.monotonic() - clock; clock = time.monotonic()
                history.append(row)
                save(policy, stats, out / 'last', **meta, history=history)
                score = row.get('held_l1', row['train_l1'])
                if best is None or score < best:
                    best = score; save(policy, stats, out / 'best', **meta, history=history)
                row['save_seconds'] = time.monotonic() - clock
                print('STEP', json.dumps(row), flush=True)
            if step >= steps:
                break
    return history


class Policy:
    """A trained network ready to answer observations."""

    def __init__(self, folder, device='cuda'):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        folder = Path(folder)
        self.meta = json.loads((folder / 'meta.json').read_text())
        self.stats = dict(np.load(folder / 'stats.npz'))
        self.device = device
        self.policy = ACTPolicy.from_pretrained(folder).to(device).eval()
        self.torch = torch

    def pictures(self, images):
        """Full-size camera pictures -> what the network was trained on (small, through JPEG), still uint8."""
        import cv2
        out = {}
        for name in CAMERAS:
            small = cv2.resize(images[name], ee_dataset.SMALL, interpolation=cv2.INTER_AREA)
            done, data = cv2.imencode('.jpg', small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, ee_dataset.JPEG_QUALITY])
            out[name] = np.ascontiguousarray(cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1])
        return out

    def predict(self, images, state):
        """Chunk of actions (chunk x 21, in the units of ee_features) for camera pictures and the 21 state values."""
        torch = self.torch
        batch = {f'observation.images.{name}': normalise(torch.as_tensor(picture)[None].to(self.device))
                 for name, picture in self.pictures(images).items()}
        batch['observation.state'] = torch.as_tensor(((np.asarray(state) - self.stats['state_mean']) / self.stats['state_std']).astype(np.float32))[None].to(self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            out = self.policy.predict_action_chunk(batch)
        return out[0].float().cpu().numpy() * self.stats['action_std'] + self.stats['action_mean']


def check(checkpoint, runs, device='cuda', batch_size=64):
    """Open-loop error of a checkpoint on the held-out runs it was not trained on."""
    from torch.utils.data import DataLoader
    policy = Policy(checkpoint, device)
    meta = policy.meta
    episodes = [Episode(path, info, meta['frame'], meta['absolute'], meta['chunk'], meta.get('paired', False)) for path, info in find_runs(runs)]
    held = [episode for episode in episodes if episode.episode in meta['held_out']]
    loader = DataLoader(Windows(held, policy.stats), batch_size=batch_size, shuffle=False, num_workers=6, worker_init_fn=quiet_worker)
    return dict(runs=len(held), frames=int(sum(e.count for e in held)), **evaluate(policy.policy, loader, policy.stats, device))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['train', 'check'])
    parser.add_argument('--runs', type=Path, required=True, help='Folder holding the run outputs with rendered datasets')
    parser.add_argument('--out', type=Path, help='train: new folder for the checkpoints')
    parser.add_argument('--checkpoint', type=Path, help='check: a checkpoint folder (best or last)')
    parser.add_argument('--held-out', type=int, nargs='*', default=list(HELD_OUT), help='Episodes kept out of training')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--frame', choices=['room', 'pelvis'], default='room')
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--independent-hands', action='store_true',
                        help='A relative action for each hand instead of the left hand and the pair (ee_features.py)')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check-every', type=int, default=1000)
    parser.add_argument('--limit-runs', type=int, help='Use only the first N runs (smoke test)')
    parser.add_argument('--device', default='cuda', help='cuda (default) or cpu')
    args = parser.parse_args()
    if args.action == 'train':
        train(args.runs, args.out, tuple(args.held_out), args.steps, args.batch_size, args.lr, args.frame, args.absolute, args.workers,
              args.seed, device=args.device, check_every=args.check_every, limit_runs=args.limit_runs,
              paired=not args.independent_hands)
    else:
        print(json.dumps(check(args.checkpoint, args.runs), indent=2))


if __name__ == '__main__':
    main()
