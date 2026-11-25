"""Train the FiSTA elbow prior on packed teleop windows (``elbow_data.py``) and judge it on held-out sessions.

Needs torch and nothing else of this package, so it runs with only ``igris/hl_ik/{common,network,train,splits}.py``
next to it.  On the A100 server it is run inside a container, on the one GPU given to that container.

  train    one seed on one device.  The network, the loss, the optimiser, the schedule and the early stopping are
           those of ``igris.hl_ik.train.train``, as used for the human-motion checkpoints; only the data differ.
  compare  on the held-out test sessions: every named checkpoint, each with its own normalisation, in metres
           and radians, next to "the elbow stays where it was" (persistence).

Swivel noise (``--swivel-noise DEG``).  FiSTA predicts the next elbow from the last five hand *and elbow* poses,
and at 30 Hz the elbow hardly moves between steps (0.4 mm on this data): trained as it is, the network learns
"the elbow stays about where it was" and predicts one step to 0.6 mm.  At run time its history is its own
output, realised by the IK.  Nothing then pulls a drifted elbow back, and on held-out teleop sessions the rollout
ended 4.1-4.8 cm from the operator's elbow, further than with no prior at all (3.8 cm).  With the noise, every
training window has the elbow history of each arm turned about the shoulder-hand line by one random angle - the
arm's one free turn with the hand held, i.e. the way a rollout is off - while the elbow to predict stays the
recorded one.  The network then has to place the elbow from the hands and can only lean on its history as far as
the history is believable.  On the validation sessions this brought the rollout to 2.3-2.6 cm, still behind the
posture lookup (1.7 cm): the turned history is not exactly a pose the arm can take, and the network can read the
turn back out of it.

Hands only (``--mask-elbow-history``).  The elbow history is the path the drift takes, so this variant does not
give it to the network at all: the elbow channels of the history are replaced by their training mean, in
training and at run time (the exported weights carry the flag).  The prediction is then a function of the last
five hand poses and the next hand goal, and a rollout cannot drift.

Swivel output (``--output swivel``).  A hands-only network still gives an elbow *place*, and for hand poses the
teleoperation never had (in simulation: the third box carried high, after the pelvis had moved) that place was
6-18 cm from anywhere the arm could put its elbow.  With the hand held the elbow has one freedom, its turn about
the shoulder-hand line.  This variant makes the network give that one angle per arm instead (last layer 1 wide,
label ``posture.swivel_deg`` of the recorded arm).  Any answer is then a posture the arm can take, and the replay
cuts it to the range the operators used (1st to 99th percentile of the training labels, kept in the checkpoint).
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path


def swivel_noise(history, sigma_rad, generator=None):
    """Copy of ``history`` (windows x 5 x 2 arms x 14, shoulder-relative) with the elbow pose of every window and
    arm turned about the shoulder-hand line by one angle drawn from N(0, sigma).  The hand pose is not changed."""
    import torch
    if sigma_rad <= 0:
        return history
    out = history.clone()
    angle = torch.randn(history.shape[0], 1, history.shape[2], 1, generator=generator, dtype=history.dtype).to(history.device) * sigma_rad
    hand = history[..., :3]; elbow = history[..., 7:10]; quaternion = history[..., 10:14]
    axis = hand / hand.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    out[..., 7:10] = elbow * cosine + torch.cross(axis, elbow, dim=-1) * sine + axis * (axis * elbow).sum(-1, keepdim=True) * (1 - cosine)
    # The elbow frame turns with it: r * q with r = (cos(a/2), axis sin(a/2)), quaternions w x y z.
    w = torch.cos(angle / 2); v = axis * torch.sin(angle / 2)
    qw, qv = quaternion[..., :1], quaternion[..., 1:]
    out[..., 10:11] = w * qw - (v * qv).sum(-1, keepdim=True)
    out[..., 11:14] = w * qv + qw * v + torch.cross(v, qv, dim=-1)
    return out


def swivel_angle(hand, elbow):
    """Swivel angle [rad] of each arm as ``posture.swivel_deg`` measures it: the elbow's turn about the shoulder-hand
    line, 0 hanging down, + outward.  ``hand`` and ``elbow`` are (..., 2 arms, 3), from the shoulder, torso axes."""
    import torch
    axis = hand / hand.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    down = torch.tensor([0., 0., -1.], dtype=hand.dtype, device=hand.device)
    reference = down - axis * (axis * down).sum(-1, keepdim=True)
    reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    radial = elbow - axis * (axis * elbow).sum(-1, keepdim=True)
    radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    side = torch.tensor([1., -1.], dtype=hand.dtype, device=hand.device)[:, None]          # left, right
    outward = torch.cross(axis, reference, dim=-1) * side
    return torch.atan2((radial * outward).sum(-1), (radial * reference).sum(-1))


def build(settings):
    """The network of a checkpoint's ``config``: FiSTA, with one output per arm instead of seven for a swivel network."""
    from torch import nn
    from igris.hl_ik.network import BimanualFiSTA, NetworkConfig
    net = BimanualFiSTA(NetworkConfig(**settings['network']))
    if settings.get('output') == 'swivel':
        for arm in net.arms:
            arm.output[-1] = nn.Linear(arm.output[-1].in_features, 1)
    return net


def judge(net, states, starts, saved, batch=4096):
    """One-step errors on recorded windows for a checkpoint dict (``mean``, ``std``, ``config`` and, for a swivel
    network, ``swivel_mean``/``swivel_std``): elbow place and frame for a pose network, swivel angle for both."""
    import numpy as np
    import torch
    from igris.hl_ik.network import batch_from_states, decode
    mean, std = saved['mean'].to(states.device), saved['std'].to(states.device)
    swivel = saved['config'].get('output') == 'swivel'; mask = saved['config'].get('elbow_history') == 'masked'
    net.eval(); total = 0.; position = []; frame = []; stay = []; turn = []; turn_stay = []
    with torch.no_grad():
        for ids in starts.split(batch):
            h, g, y, raw = batch_from_states(states, ids, mean, std)
            if mask:
                h[..., 7:] = 0.
            future = states[ids + 5]; last = states[ids + 4]
            truth = swivel_angle(future[:, :, :3], future[:, :, 7:10])
            prediction = net(h, g)
            if swivel:
                wanted = (truth - saved['swivel_mean'].to(states.device)) / saved['swivel_std'].to(states.device)
                total += float((prediction[:, :, 0] - wanted).square().sum())
                angle = prediction[:, :, 0] * saved['swivel_std'].to(states.device) + saved['swivel_mean'].to(states.device)
            else:
                total += float((prediction - y).square().sum())
                pose = decode(prediction, mean, std)
                position.append((pose[:, :, :3] - raw[:, :, :3]).norm(dim=-1).cpu().numpy())
                unit = raw[:, :, 3:] / raw[:, :, 3:].norm(dim=-1, keepdim=True)
                frame.append((2 * torch.acos((pose[:, :, 3:] * unit).sum(-1).abs().clamp(max=1))).cpu().numpy())
                angle = swivel_angle(future[:, :, :3], pose[:, :, :3])           # the swivel its elbow place stands for
            difference = torch.atan2(torch.sin(angle - truth), torch.cos(angle - truth)).abs()
            turn.append(torch.rad2deg(difference).cpu().numpy())
            held = swivel_angle(last[:, :, :3], last[:, :, 7:10])
            turn_stay.append(torch.rad2deg(torch.atan2(torch.sin(held - truth), torch.cos(held - truth)).abs()).cpu().numpy())
            stay.append((last[:, :, 7:10] - raw[:, :, :3]).norm(dim=-1).cpu().numpy())
    turn, turn_stay, stay = np.concatenate(turn), np.concatenate(turn_stay), np.concatenate(stay)
    report = dict(normalized_mse=total / (len(starts) * (2 if swivel else 14)), samples=len(starts),
                  swivel_error_mean_deg=float(turn.mean()), swivel_error_p95_deg=float(np.percentile(turn, 95)),
                  persistence_swivel_mean_deg=float(turn_stay.mean()), persistence_position_mean_m=float(stay.mean()))
    if not swivel:
        position, frame = np.concatenate(position), np.concatenate(frame)
        report.update(position_mean_m=float(position.mean()), position_p95_m=float(np.percentile(position, 95)),
                      position_rmse_m=float(np.sqrt((position * position).mean())),
                      orientation_mean_rad=float(frame.mean()), orientation_p95_rad=float(np.percentile(frame, 95)))
    return report


def train(root, scope, seed, device, epochs=60, batch=2048, swivel_noise_deg=0., variant=None, mask_elbow_history=False,
          output='pose'):
    """Train one seed on ``root/packed/<scope>``; the checkpoint goes to ``root/training/<variant or scope>/seed_N``.

    Without swivel noise, masking or a swivel output this is ``igris.hl_ik.train.train`` itself.  Otherwise the same
    loop runs here, with the elbow history of the training windows turned or masked, or with the network giving the
    swivel angle of each arm instead of its elbow pose (``output='swivel'``, always hands only; see the module text)."""
    from igris.hl_ik import train as hl_train
    root = Path(root)
    if output not in ('pose', 'swivel'):
        raise ValueError(output)
    if output == 'swivel':
        mask_elbow_history = True
    if not swivel_noise_deg and not mask_elbow_history:
        if variant not in (None, scope):
            raise ValueError('A variant name needs the swivel noise or the masking it stands for')
        hl_train.train(scope, seed, epochs, batch, device, output=root)
        return root / 'training' / scope / f'seed_{seed}' / 'best.pt'
    import numpy as np
    import torch
    from igris.hl_ik.common import save_json, sha256
    from igris.hl_ik.network import NetworkConfig, canonicalize
    variant = variant or (f'{scope}_swivel' if output == 'swivel' else f'{scope}_hands' if mask_elbow_history
                          else f'{scope}_s{swivel_noise_deg:g}')
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    sigma = math.radians(swivel_noise_deg)
    train_s, train_i = hl_train.load_split(scope, 'train', device, root)
    val_s, val_i = hl_train.load_split(scope, 'val', device, root)
    mean, std = hl_train.stats(train_s, train_i)                         # of the recorded windows, without noise
    out = root / 'training' / variant / f'seed_{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists():
        raise FileExistsError(f'checkpoint exists: {out}; use a new variant name instead of overwriting')
    settings = dict(scope=scope, variant=variant, seed=seed, epochs=epochs, batch=batch, optimizer='AdamW', lr=.001,
                    weight_decay=.0001, schedule='cosine to 1e-5', early_stopping_patience=10,
                    normalization='training windows only, per-arm standardization with std floor .05',
                    loss='coordinate MSE on normalized xyz/wxyz; window hemisphere alignment' if output == 'pose'
                         else 'MSE on the standardized swivel angle of each arm',
                    checkpoint_choice='lowest loss on the validation sessions with the same swivel noise (fixed draw) and masking',
                    output=output, elbow_history='masked' if mask_elbow_history else 'given',
                    history_swivel_noise_deg=float(swivel_noise_deg),
                    history_swivel_noise='elbow history of every training window and arm turned about the shoulder-hand '
                                         'line by one angle from N(0, sigma); the label is the recorded elbow',
                    network=asdict(NetworkConfig()), device=device,
                    data_manifest_sha256=sha256(root / 'packed' / scope / 'manifest.json'))
    net = build(settings).to(device)
    settings['parameter_count'] = sum(p.numel() for p in net.parameters())
    extra = {}
    if output == 'swivel':
        # Label statistics and the range the operators used, from the training windows.
        labels = torch.cat([swivel_angle(train_s[ids + 5][:, :, :3], train_s[ids + 5][:, :, 7:10]) for ids in train_i.split(65536)])
        extra = dict(swivel_mean=labels.mean(0).cpu(), swivel_std=labels.std(0).clamp_min(.05).cpu(),
                     swivel_range=torch.stack([torch.quantile(labels[::7, arm], torch.tensor([.01, .99], device=labels.device))
                                               for arm in range(2)]).cpu())
        settings['swivel_range_deg'] = torch.rad2deg(extra['swivel_range']).tolist()
    save_json(out / 'config.json', settings)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-5)
    offsets = torch.arange(5, device=train_s.device)[None]

    def windows(states, ids, generator=None):
        past = swivel_noise(states[ids[:, None] + offsets], sigma, generator); future = states[ids + 5]
        h, g, y = canonicalize(past, future[:, :, :7], future[:, :, 7:])
        h = (h - mean) / std
        if mask_elbow_history:
            h[..., 7:] = 0.                                              # the training mean, in normalised units
        if output == 'swivel':
            angle = swivel_angle(future[:, :, :3], future[:, :, 7:10])
            return h, (g - mean[:, :7]) / std[:, :7], ((angle - extra['swivel_mean'].to(angle.device)) / extra['swivel_std'].to(angle.device))[..., None]
        return h, (g - mean[:, :7]) / std[:, :7], (y - mean[:, 7:]) / std[:, 7:]

    @torch.no_grad()
    def noisy_validation():
        """The training objective on the validation sessions, with the same noise for every epoch."""
        net.eval(); generator = torch.Generator().manual_seed(0); total = 0.; count = 0
        for ids in val_i.split(4096):
            h, g, y = windows(val_s, ids, generator)
            total += float((net(h, g) - y).square().sum()); count += y.numel()
        return total / count

    def checkpoint(epoch, validation):
        return dict(network=net.state_dict(), config=settings, mean=mean.cpu(), std=std.cpu(), epoch=epoch, validation=validation, **extra)

    best = float('inf'); wait = 0; history = []; start = time.monotonic()
    for epoch in range(1, epochs + 1):
        net.train(); total = 0.
        for ids in train_i[torch.randperm(len(train_i), device=train_i.device)].split(batch):
            h, g, y = windows(train_s, ids)
            optimiser.zero_grad(set_to_none=True)
            loss = (net(h, g) - y).square().mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimiser.step(); total += float(loss.detach()) * len(ids)
        schedule.step()
        metrics = judge(net, val_s, val_i, checkpoint(epoch, None))      # recorded windows, for comparison with other runs
        score = noisy_validation()                                       # what this run is trained for: chosen on this
        row = dict(epoch=epoch, train_mse=total / len(train_i), noisy_validation_mse=score, **metrics,
                   elapsed_s=time.monotonic() - start)
        history.append(row); save_json(out / 'history.json', history)
        if score < best - 1e-6:
            best = score; wait = 0
            torch.save(checkpoint(epoch, dict(metrics, noisy_validation_mse=score)), out / 'best.pt')
        else:
            wait += 1
        print(json.dumps(dict(variant=variant, seed=seed, **row)), flush=True)
        if wait >= 10:
            break
    saved = torch.load(out / 'best.pt', map_location=device, weights_only=False); net.load_state_dict(saved['network'])
    test_s, test_i = hl_train.load_split(scope, 'test', device, root)
    test = judge(net, test_s, test_i, saved)
    save_json(out / 'test_own_split.json', test)
    save_json(out / 'complete.json', dict(best_epoch=saved['epoch'], epochs_run=len(history), elapsed_s=time.monotonic() - start, test=test))
    return out / 'best.pt'


def compare(root, scope, checkpoints, device, split='test'):
    """Metrics of each checkpoint (name -> path) on one packed split of ``root`` (recorded windows, no noise)."""
    import torch
    from igris.hl_ik import train as hl_train
    states, starts = hl_train.load_split(scope, split, device, Path(root))
    report = {}
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location=device, weights_only=False)
        net = build(saved['config']).to(device)
        net.load_state_dict(saved['network'])
        report[name] = dict(judge(net, states, starts, saved), checkpoint=str(path), best_epoch=int(saved['epoch']),
                            elbow_history_masked=saved['config'].get('elbow_history') == 'masked',
                            output=saved['config'].get('output', 'pose'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['train', 'compare'])
    parser.add_argument('--root', type=Path, required=True, help='Experiment root holding packed/<scope>/')
    parser.add_argument('--scope', default='teleop')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--swivel-noise', type=float, default=0., metavar='DEG',
                        help='train: turn the elbow history of the training windows by N(0, DEG) about the shoulder-hand line')
    parser.add_argument('--mask-elbow-history', action='store_true',
                        help='train: hands only - the network is not given the elbow history (variant <scope>_hands)')
    parser.add_argument('--output', choices=['pose', 'swivel'], default='pose',
                        help='train: what the network gives per arm - the elbow pose (FiSTA) or its swivel angle (hands only)')
    parser.add_argument('--variant', help='train: folder name under training/ (default <scope>_s<DEG>, <scope>_hands or <scope>_swivel)')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH', help='compare: a checkpoint to judge')
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', type=Path, help='compare: where to write the report (JSON)')
    args = parser.parse_args()
    if args.command == 'train':
        print('BEST', train(args.root, args.scope, args.seed, args.device, args.epochs, args.batch, args.swivel_noise, args.variant,
                            args.mask_elbow_history, args.output), flush=True)
    else:
        named = dict(item.split('=', 1) for item in args.checkpoint)
        if not named:
            parser.error('compare needs at least one --checkpoint NAME=PATH')
        report = compare(args.root, args.scope, named, args.device, args.split)
        text = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(text + '\n')
        print(text)


if __name__ == '__main__':
    main()
