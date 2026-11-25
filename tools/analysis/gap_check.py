"""How much do the two hands squeeze a box, and how well would independent hand predictions keep that?"""
import numpy as np, sys, json
from igris.umi_walklow_replay import act_ee, ee_features
gap_cmd=[]; gap_meas=[]; held=[]
for path, info in act_ee.find_runs(sys.argv[1])[:30]:
    d = np.load(path / 'dataset/frames.npz')
    cmd = np.linalg.norm(d['target_pelvis'][:, 0, :3, 3] - d['target_pelvis'][:, 1, :3, 3], axis=1)
    meas = np.linalg.norm(d['ee_pelvis'][:, 0, :3, 3] - d['ee_pelvis'][:, 1, :3, 3], axis=1)
    z = d['parcels_room'][:, :, 2]; lifted = (z - z[0]).max(axis=1) > .03          # some parcel is in the air
    moving = np.abs(np.gradient(z, axis=0)).max(axis=1) * 15 > .02
    carry = lifted & moving
    gap_cmd.append(cmd[carry]); gap_meas.append(meas[carry])
    held.append(carry.mean())
c = np.concatenate(gap_cmd); m = np.concatenate(gap_meas)
print('frames with a box moving in the air:', round(float(np.mean(held)), 3))
print('hand-to-hand distance while carrying [mm]: commanded p5/p50/p95', np.percentile(c, [5, 50, 95]).round(4) * 1e3, ' measured', (np.percentile(m, [5, 50, 95]) * 1e3).round(1))
print('measured minus commanded (the squeeze the commanded pair asks for) [mm]: p5/p50/p95', (np.percentile(m - c, [5, 50, 95]) * 1e3).round(1))
