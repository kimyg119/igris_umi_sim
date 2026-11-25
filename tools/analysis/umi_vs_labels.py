"""UMI as recorded (constants only: hand calibration, palm -> EE frame) against the labels the robot followed,
both as relative EE actions over the same recorded interval.  Offline, numpy."""
import glob, json, os
import numpy as np
from igris.umi_walklow_replay import ee_features, retarget
from igris.umi_walklow_replay.governor import blend_pose
from igris.umi_walklow_replay.kinematics import rotation_vector

def poses(rows):
    return retarget.pose_matrices(rows[:, :7]), retarget.pose_matrices(rows[:, 7:14])

def at(stamps, series, wanted):
    index = np.clip(np.searchsorted(stamps, wanted, side='right') - 1, 0, len(stamps) - 2)
    share = np.clip((wanted - stamps[index]) / (stamps[index + 1] - stamps[index]), 0., 1.)
    return np.array([blend_pose(series[i], series[i + 1], w) for i, w in zip(index, share)])

def angle(a, b):
    return np.degrees([np.linalg.norm(rotation_vector(x[:3, :3].T @ y[:3, :3])) for x, y in zip(a, b)])

rows = {key: [] for key in ('move', 'turn', 'gap', 'pair_turn', 'held', 'size')}
for path in sorted(glob.glob(os.environ.get('UMI_LABELS', 'outputs/umi_pose_labels/labels') + '/episode_*.npz')):
    d = np.load(path)
    left, right = poses(d['state']); source = d['state_source_t']
    raw_left, raw_right = poses(d['raw_state'])
    raw_l = at(d['raw_t'], raw_left, source); raw_r = at(d['raw_t'], raw_right, source)
    held = np.zeros(len(source), dtype=bool)
    for first, last in d['carry_frames']:
        held |= (source >= first / float(d['fps']) / float(d['speed'])) & (source <= last / float(d['fps']) / float(d['speed']))
    step = 4                                                         # 4 samples of the label clock: one replanning period
    a = ee_features.inverse(left[:-step]) @ left[step:]; b = ee_features.inverse(raw_l[:-step]) @ raw_l[step:]
    rows['move'].append(np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)); rows['turn'].append(angle(a, b))
    rows['size'].append(np.linalg.norm(a[:, :3, 3], axis=1))
    pa = ee_features.inverse(left) @ right; pb = ee_features.inverse(raw_l) @ raw_r
    rows['gap'].append((np.linalg.norm(pb[:, :3, 3], axis=1) - np.linalg.norm(pa[:, :3, 3], axis=1))[:-step])
    rows['pair_turn'].append(angle(pa, pb)[:-step]); rows['held'].append(held[:-step])
held = np.concatenate(rows['held'])
def line(name, key, unit, scale=1.):
    v = np.concatenate(rows[key]) * scale
    for label, mask in (('box held', held), ('hands free', ~held)):
        print(f'{name:58s} {label:10s} p50 {np.percentile(v[mask], 50):7.1f}  p95 {np.percentile(v[mask], 95):7.1f} {unit}')
print('episodes', len(rows['move']), 'samples', len(held), 'share with a box held', round(float(held.mean()), 2))
line('left hand move over 0.27 s (label)', 'size', 'mm', 1e3)
line('  ... label minus recording, same interval', 'move', 'mm', 1e3)
line('  ... rotation difference', 'turn', 'deg')
line('hand-to-hand distance: recording minus label', 'gap', 'mm', 1e3)
line('pair orientation: recording against label', 'pair_turn', 'deg')
