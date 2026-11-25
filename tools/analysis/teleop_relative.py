"""The teleoperated joint recordings as the same relative EE actions (joints -> links -> EE), next to the UMI labels."""
import glob, json, os
import numpy as np
import pyarrow.parquet as pq
from igris.umi_walklow_replay import ee_features, retarget, teleop, act_ee
from igris.umi_walklow_replay.kinematics import UpperBody, rotation_vector

root = os.environ.get('UMI_TELEOP_DATASET', 'datasets/Igris_Teleop_Box_Palletizing_full_signals_15hz')
info = json.load(open(root + '/meta/info.json'))
tasks = pq.read_table(root + '/meta/tasks.parquet').to_pylist()
print('teleop:', info['total_episodes'], 'episodes,', info['total_frames'], 'frames,', info['fps'], 'Hz; tasks:', [str(next(iter(t.values())) if not any(isinstance(v, str) for v in t.values()) else [v for v in t.values() if isinstance(v, str)][0])[:70] for t in tasks][:7])
model = UpperBody.from_mjcf()

def fk(q):
    hands = np.array([model.fk(row) for row in q]); chest = np.array([model.torso(row) for row in q])
    return hands, chest

def describe(name, now, chest, waist_now, targets, waist_targets):
    state = ee_features.state(now, chest, waist_now)
    actions, pad = act_ee.chunks(now, waist_now, targets, waist_targets, length=8)
    one = np.linalg.norm(actions[:, 0, :3], axis=1) * 1e3                     # target of a frame from the hand of that frame
    step = np.linalg.norm(np.diff(targets[:, 0, :3, 3], axis=0), axis=1) * 1e3
    turn = np.degrees([np.linalg.norm(rotation_vector(a[:3, :3].T @ b[:3, :3])) for a, b in zip(targets[:-1, 0], targets[1:, 0])])
    gap = np.linalg.norm(actions[:, 0, 9:12], axis=1) * 1e3
    return dict(name=name, frames=len(now), one=one, step=step, turn=turn, gap=gap, state=state, waist=np.degrees(waist_targets),
                waist_step=np.degrees(np.abs(np.diff(waist_targets, axis=0)).max(axis=1)), chunk=actions[~pad[:, 7], 7])

sets = {'teleop (command vs measured joints)': [], 'UMI labels (robot followed them)': []}
rng = np.random.default_rng(0)
for episode in sorted(rng.choice(info['total_episodes'], 60, replace=False).tolist()):
    try:
        data = teleop.load_episode(root, episode)
    except Exception as error:
        print('skip', episode, str(error)[:80]); continue
    now, chest = fk(data['state']); targets, _ = fk(data['action'])
    sets['teleop (command vs measured joints)'].append(describe(str(episode), now, chest, data['state'][:, :3], targets, data['action'][:, :3]))
for path in sorted(glob.glob(os.environ.get('UMI_LABELS', 'outputs/umi_pose_labels/labels') + '/episode_*.npz')):
    d = np.load(path); q = d['state_q']
    hands, chest = fk(q)                                              # the label joint path: target and (ideal) state alike
    sets['UMI labels (robot followed them)'].append(describe(path[-7:-4], hands, chest, q[:, :3], np.roll(hands, -1, axis=0), np.roll(q[:, :3], -1, axis=0)))
def cat(rows, key): return np.concatenate([row[key] for row in rows])
for name, rows in sets.items():
    print(f'\n== {name}: {len(rows)} episodes, {sum(r["frames"] for r in rows)} frames')
    for key, label, unit in (('one', 'target of a frame from the hand of that frame', 'mm'), ('step', 'left target step per sample', 'mm'), ('turn', 'left target turn per sample', 'deg'),
                             ('gap', 'right target from left target (pair distance)', 'mm'), ('waist_step', 'waist target step per sample', 'deg')):
        v = cat(rows, key); print(f'   {label:48s} p5 {np.percentile(v,5):7.1f}  p50 {np.percentile(v,50):7.1f}  p95 {np.percentile(v,95):7.1f}  p99 {np.percentile(v,99):7.1f} {unit}')
    w = cat(rows, 'waist'); print('   waist target range [deg] yaw/roll/pitch p1..p99:', [(round(float(np.percentile(w[:, i], 1)), 1), round(float(np.percentile(w[:, i], 99)), 1)) for i in range(3)])
    s = cat(rows, 'state'); print('   left hand in the chest frame  p1..p99 [m]: x', np.percentile(s[:, 0], [1, 99]).round(2), 'y', np.percentile(s[:, 1], [1, 99]).round(2), 'z', np.percentile(s[:, 2], [1, 99]).round(2))
    print('   right hand in the chest frame p1..p99 [m]: x', np.percentile(s[:, 9], [1, 99]).round(2), 'y', np.percentile(s[:, 10], [1, 99]).round(2), 'z', np.percentile(s[:, 11], [1, 99]).round(2))
# how much of one set's action distribution lies inside the other's 1..99 % box, per action value (8th step of the chunk: 0.53 s ahead)
a = cat(sets['teleop (command vs measured joints)'], 'chunk'); b = cat(sets['UMI labels (robot followed them)'], 'chunk')
lo, hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
inside = ((b >= lo) & (b <= hi)).mean(axis=0)
print('\nshare of UMI-label actions (0.53 s ahead) inside the 1-99 % range of the teleop actions, per value:')
print('   left hand xyz', inside[:3].round(2), 'rotation', inside[3:9].round(2).tolist(), '| pair xyz', inside[9:12].round(2), 'rotation', inside[12:18].round(2).tolist(), '| waist', inside[18:].round(2))
