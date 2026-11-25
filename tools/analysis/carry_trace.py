"""The third carry of one run, second by second: targets, IK error, the gap between the hands, the box, the pelvis."""
import sys, json
import numpy as np
from igris.umi_walklow_replay.analyze import tilt_deg, yaw
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody
model = UpperBody.from_mjcf(); lower = model.lower[ACTIVE] + .035; upper = model.upper[ACTIVE] - .035
log = np.load(sys.argv[1] + '/run/replay_log.npz'); m = log['phase'] == 'replay'
sim = log['sim'][m] - log['sim'][m][0]; box = log['parcels'][m][:, 0]
t0 = float(sys.argv[2]); t1 = float(sys.argv[3])
gap_cmd = np.linalg.norm(log['target_left_pose'][m][:, :3, 3] - log['target_right_pose'][m][:, :3, 3], axis=1)
gap_sim = np.linalg.norm(log['hands'][m][:, 0, :3] - log['hands'][m][:, 1, :3], axis=1)
turn = np.degrees(np.unwrap(yaw(log['pelvis'][m][:, 3:]))); turn -= turn[0]
q = log['q_command'][m][:, ACTIVE]
for t in np.arange(t0, t1, float(sys.argv[4]) if len(sys.argv) > 4 else 1.):
    i = min(np.searchsorted(sim, t), len(sim) - 1)
    at = [JOINTS[ACTIVE[j]] for j in range(17) if q[i, j] <= lower[j] + 1e-4 or q[i, j] >= upper[j] - 1e-4]
    print(f't={t:5.1f} target z L/R {log["target_left_pose"][m][i,2,3]:.2f}/{log["target_right_pose"][m][i,2,3]:.2f} ik err L/R {log["position_error"][m][i,0]*1e3:5.1f}/{log["position_error"][m][i,1]*1e3:5.1f} mm '
          f'gap cmd {gap_cmd[i]*1e3:4.0f} sim {gap_sim[i]*1e3:4.0f} box z {box[i,2]:.2f} tilt {tilt_deg(box[i,3:]):4.0f} waist {np.degrees(q[i,:3]).round(0)} pelvis turn {turn[i]:5.1f} at limit {at}')
