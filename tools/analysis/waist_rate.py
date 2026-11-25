"""Does a fast waist turn the pelvis?  Per run: how fast the waist yaw was commanded and how far the pelvis turned."""
import json, sys
from pathlib import Path
import numpy as np
from igris.umi_walklow_replay.analyze import yaw
for spec in sys.argv[1:]:
    name, folder = spec.split('=')
    rows = []
    for run in sorted(Path(folder).glob('ep*/run')):
        if not (run / 'replay_log.npz').exists(): continue
        log = np.load(run / 'replay_log.npz'); m = log['phase'] == 'replay'
        if m.sum() < 500: continue
        sim = log['sim'][m]; q = log['q_command'][m]; dt = np.maximum(np.diff(sim), 1e-4)
        rate = np.abs(np.diff(q[:, 0]) / dt); accel = np.abs(np.diff(np.diff(q[:, 0]) / dt) / dt[1:])
        turn = np.degrees(np.unwrap(yaw(log['pelvis'][m][:, 3:]))); turn -= turn[0]
        swing = np.abs(np.diff(q[:, 0])).sum()                   # total waist yaw travelled [rad]
        rows.append((np.percentile(rate, 99), rate.mean(), np.percentile(accel, 99), swing, np.abs(turn).max(), sim[-1] - sim[0]))
    rows = np.array(rows)
    print(f'{name:22s} runs {len(rows):3d}  waist yaw rate p99 median {np.median(rows[:,0]):.2f} rad/s  mean rate {np.median(rows[:,1]):.3f}  accel p99 {np.median(rows[:,2]):5.1f}  '
          f'yaw travelled {np.median(rows[:,3]):5.1f} rad  pelvis turn max median {np.median(rows[:,4]):5.1f} deg  seconds {np.median(rows[:,5]):.0f}  '
          f'corr(rate p99, turn) {np.corrcoef(rows[:,0], rows[:,4])[0,1]:.2f}  corr(travel, turn) {np.corrcoef(rows[:,3], rows[:,4])[0,1]:.2f}')
