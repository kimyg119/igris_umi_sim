"""When does the pelvis turn, and what is the waist doing then?  Policy run vs the demonstration of the same episode."""
import sys, json
import numpy as np
from igris.umi_walklow_replay.analyze import yaw

def trace(path, label):
    d = np.load(path + '/run/replay_log.npz'); m = d['phase'] == 'replay'
    sim = d['sim'][m] - d['sim'][m][0]; q = d['q_command'][m]; pelvis = d['pelvis'][m]; feet = d['feet'][m]
    turn = np.degrees(np.unwrap(yaw(pelvis[:, 3:]))); turn -= turn[0]
    dt = np.maximum(np.diff(sim), 1e-4); rate = np.diff(q[:, 0]) / dt; accel = np.abs(np.diff(rate)) / dt[1:]
    arm_accel = np.abs(np.diff(np.diff(q[:, 15:29], axis=0) / dt[:, None], axis=0)) / dt[1:, None]
    print(f'== {label}: waist yaw accel p50 {np.percentile(accel,50):.1f} p99 {np.percentile(accel,99):.1f} max {accel.max():.0f} rad/s2; '
          f'arm accel p99 {np.percentile(arm_accel.max(axis=1),99):.1f}; waist yaw rate p99 {np.percentile(np.abs(rate),99):.2f} rad/s')
    for t in range(0, int(sim[-1]) + 1, 8):
        i = min(np.searchsorted(sim, t), len(sim) - 1)
        print(f'   t={t:3d}s pelvis turn {turn[i]:7.1f} deg  waist yaw {np.degrees(q[i,0]):6.1f}  pelvis xy {pelvis[i,:2].round(2)} z {pelvis[i,2]:.3f} parcels z {d["parcels"][m][i,:,2].round(2)}')
for item in sys.argv[1:]:
    label, path = item.split('=')
    trace(path, label)
