"""How far off is a perfect policy?  Room-relative labels executed in the pelvis frame, per chunk step."""
import numpy as np, json, sys
from igris.umi_walklow_replay import act_ee, ee_features

rows = {k: [] for k in (2, 4, 6, 10, 16, 31)}; lag = []; step = []; waist = []
runs = act_ee.find_runs(sys.argv[1])
for path, info in runs:
    d = np.load(path / 'dataset/frames.npz')
    T = len(d['t'])
    for k in rows:
        t0 = np.arange(0, T - k)
        relative = ee_features.inverse(d['ee_room'][t0]) @ d['target_room'][t0 + k]            # the label
        executed = d['ee_pelvis'][t0] @ relative                                                # composed with the pelvis-frame hands
        rows[k].append(np.linalg.norm(executed[..., :3, 3] - d['target_pelvis'][t0 + k][..., :3, 3], axis=-1).ravel())
    lag.append(np.linalg.norm(d['target_pelvis'][:, :, :3, 3] - d['ee_pelvis'][:, :, :3, 3], axis=-1).ravel())      # target of frame k vs hands at frame k
    step.append(np.linalg.norm(np.diff(d['target_room'][:, :, :3, 3], axis=0), axis=-1).ravel())
    waist.append(np.degrees(np.abs(d['waist_target'] - d['q_measured'][:, :3])).max(axis=1))
print('runs', len(runs))
for k, values in rows.items():
    v = np.concatenate(values); print(f'chunk step {k:2d} ({k/15:.2f} s): pelvis moved meanwhile -> target off by  p50 {np.percentile(v,50)*1e3:5.1f} mm  p95 {np.percentile(v,95)*1e3:5.1f} mm  p99 {np.percentile(v,99)*1e3:5.1f} mm  max {v.max()*1e3:5.1f} mm')
v = np.concatenate(lag); print(f'one-step relative action size (target of a frame from the hands of that frame): p50 {np.percentile(v,50)*1e3:.1f} mm p95 {np.percentile(v,95)*1e3:.1f} mm max {v.max()*1e3:.1f} mm')
v = np.concatenate(step); print(f'target step per sample: p50 {np.percentile(v,50)*1e3:.1f} mm p95 {np.percentile(v,95)*1e3:.1f} mm max {v.max()*1e3:.1f} mm')
v = np.concatenate(waist); print(f'waist target minus waist now (largest joint): p50 {np.percentile(v,50):.2f} deg p95 {np.percentile(v,95):.2f} deg max {v.max():.2f} deg')
