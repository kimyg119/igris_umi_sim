"""How steady is the commanded pair (right hand target in the left target's frame) while a box is held?"""
import sys
import numpy as np
from igris.umi_walklow_replay import ee_features
from igris.umi_walklow_replay.kinematics import rotation_vector
for spec in sys.argv[1:]:
    name, path, t0, t1 = spec.split('=')[0], spec.split('=')[1].split(':')[0], *map(float, spec.split(':')[1:])
    log = np.load(path + '/run/replay_log.npz'); m = log['phase'] == 'replay'
    sim = log['sim'][m] - log['sim'][m][0]; keep = (sim >= t0) & (sim <= t1)
    left = log['target_left_pose'][m][keep]; right = log['target_right_pose'][m][keep]
    pair = ee_features.inverse(left) @ right
    turn = np.degrees([np.linalg.norm(rotation_vector(pair[0, :3, :3].T @ p[:3, :3])) for p in pair])
    shift = np.linalg.norm(pair[:, :3, 3] - pair[0, :3, 3], axis=1) * 1e3
    lateral = np.abs(pair[:, :3, 3] - pair[0, :3, 3]) * 1e3
    lz = left[:, 2, 3]; speed = np.abs(np.diff(left[:, :3, 3], axis=0)).sum(axis=1) / np.maximum(np.diff(sim[keep]), 1e-4)
    accel = np.abs(np.diff(np.diff(left[:, :3, 3], axis=0) / np.maximum(np.diff(sim[keep]), 1e-4)[:, None], axis=0)).max(axis=1) / np.maximum(np.diff(sim[keep])[1:], 1e-4)
    hand_turn = np.degrees([np.linalg.norm(rotation_vector(left[0, :3, :3].T @ p[:3, :3])) for p in left])
    print(f'{name:14s} {t0:.0f}-{t1:.0f} s: pair turned from its start by max {turn.max():4.1f} deg (p50 {np.median(turn):.1f}), pair position moved max {shift.max():5.1f} mm (per axis {lateral.max(axis=0).round(1)}); '
          f'left target rose {1e2*(lz.max()-lz[0]):.0f} cm, turned {hand_turn.max():.0f} deg, speed p99 {np.percentile(speed,99):.2f} m/s, accel p99 {np.percentile(accel,99):.1f} m/s2')
