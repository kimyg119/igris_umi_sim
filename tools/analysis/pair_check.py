"""Open-loop check of a checkpoint on its held-out runs: how well is the hand gap (the squeeze) predicted while a box is carried?"""
import sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
from igris.umi_walklow_replay import act_ee, ee_features

policy = act_ee.Policy(sys.argv[1]); meta = policy.meta
found = [(p, i) for p, i in act_ee.find_runs(meta['runs'])]
rows = []
for path, info in found:
    episode = act_ee.Episode(path, info, meta['frame'], meta['absolute'], meta['chunk'], meta.get('paired', False))
    if episode.episode not in meta['held_out']:
        continue
    data = np.load(path / 'dataset/frames.npz')
    z = data['parcels_room'][:, :, 2]; carry = ((z - z[0]).max(axis=1) > .03) & (np.abs(np.gradient(z, axis=0)).max(axis=1) * 15 > .02)
    windows = act_ee.Windows([episode], policy.stats)
    loader = DataLoader(windows, batch_size=128, shuffle=False, num_workers=4, worker_init_fn=act_ee.quiet_worker)
    predicted = []
    with torch.no_grad():
        for batch in loader:
            batch = act_ee.to_device(batch, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = policy.policy.predict_action_chunk(batch)
            predicted.append(out.float().cpu().numpy() * policy.stats['action_std'] + policy.stats['action_mean'])
    predicted = np.concatenate(predicted); wanted = episode.actions
    steps = slice(2, 10)
    if meta.get('paired', False):
        gap_p = np.linalg.norm(predicted[:, steps, 9:12], axis=-1); gap_w = np.linalg.norm(wanted[:, steps, 9:12], axis=-1)
    else:   # a relative action per hand: rebuild both targets from the hands of the frame
        hands = data[f"ee_{meta['frame']}"]
        gap_p = np.empty((len(hands), 8)); gap_w = np.empty((len(hands), 8))
        for t in range(len(hands)):
            a, _ = ee_features.absolute_targets(hands[t], np.zeros(3), predicted[t, steps], False); b, _ = ee_features.absolute_targets(hands[t], np.zeros(3), wanted[t, steps], False)
            gap_p[t] = np.linalg.norm(a[:, 0, :3, 3] - a[:, 1, :3, 3], axis=-1); gap_w[t] = np.linalg.norm(b[:, 0, :3, 3] - b[:, 1, :3, 3], axis=-1)
    left = np.linalg.norm(predicted[:, steps, :3] - wanted[:, steps, :3], axis=-1)
    rows.append(dict(run=path.name, gap_carry=np.abs(gap_p - gap_w)[carry].ravel(), gap_free=np.abs(gap_p - gap_w)[~carry].ravel(),
                     left_carry=left[carry].ravel(), left_free=left[~carry].ravel(), gap_signed=(gap_p - gap_w)[carry].ravel()))
def stats(key):
    v = np.concatenate([row[key] for row in rows]) * 1e3
    return dict(mean=round(float(np.mean(v)), 1), p95=round(float(np.percentile(v, 95)), 1), max=round(float(np.max(v)), 1))
print(json.dumps(dict(checkpoint=sys.argv[1], runs=len(rows), paired=meta.get('paired', False),
                      hand_gap_error_while_carrying_mm=stats('gap_carry'), hand_gap_error_otherwise_mm=stats('gap_free'),
                      left_hand_error_while_carrying_mm=stats('left_carry'), left_hand_error_otherwise_mm=stats('left_free'),
                      gap_too_wide_by_mm_mean=round(float(np.concatenate([r['gap_signed'] for r in rows]).mean() * 1e3), 1)), indent=1))
