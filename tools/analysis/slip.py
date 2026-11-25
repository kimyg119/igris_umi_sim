"""Does a carried box move in the hands?  Box centre in the frame of the left hand (simulator poses), over the carry."""
import sys
import numpy as np
from igris.umi_walklow_replay.worker import pose_matrix
from igris.umi_walklow_replay.analyze import tilt_deg
for spec in sys.argv[1:]:
    name, path, index, t0, t1 = spec.split('=')[0], *spec.split('=')[1].split(':')
    index = int(index); t0 = float(t0); t1 = float(t1)
    log = np.load(path + '/run/replay_log.npz'); m = log['phase'] == 'replay'; sim = log['sim'][m] - log['sim'][m][0]
    print(name)
    first = None
    for t in np.arange(t0, t1, 1.):
        i = min(np.searchsorted(sim, t), len(sim) - 1)
        left = pose_matrix(log['hands'][m][i, 0]); right = pose_matrix(log['hands'][m][i, 1]); box = pose_matrix(log['parcels'][m][i, index])
        in_left = (np.linalg.inv(left) @ box)[:3, 3]; mid = (left[:3, 3] + right[:3, 3]) / 2
        if first is None: first = in_left
        print(f'   t={t:5.1f} box in the left hand frame {np.round(in_left*1e3).astype(int)} mm (moved {np.linalg.norm(in_left-first)*1e3:5.1f}), box z {box[2,3]:.3f}, hands mid z {mid[2]:.3f}, box below hands {1e3*(mid[2]-box[2,3]):5.0f} mm, hands gap {np.linalg.norm(left[:3,3]-right[:3,3])*1e3:.0f}, tilt {tilt_deg(log["parcels"][m][i,index,3:]):.0f}')
