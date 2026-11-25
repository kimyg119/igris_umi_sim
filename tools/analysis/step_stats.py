"""How fast do the demonstrated hand targets move per stream sample (15 Hz)?  Left target, in the pelvis frame as commanded."""
import sys
import numpy as np
from igris.umi_walklow_replay import act_ee, ee_features
from igris.umi_walklow_replay.kinematics import rotation_vector
move = []; turn = []; pair_move = []; pair_turn = []; waist = []
for path, info in act_ee.find_runs(sys.argv[1]):
    d = np.load(path / 'dataset/frames.npz'); left = d['target_pelvis'][:, 0]; right = d['target_pelvis'][:, 1]
    move.append(np.linalg.norm(np.diff(left[:, :3, 3], axis=0), axis=1))
    turn.append([np.linalg.norm(rotation_vector(a[:3, :3].T @ b[:3, :3])) for a, b in zip(left[:-1], left[1:])])
    pair = ee_features.inverse(left) @ right
    pair_move.append(np.linalg.norm(np.diff(pair[:, :3, 3], axis=0), axis=1))
    pair_turn.append([np.linalg.norm(rotation_vector(a[:3, :3].T @ b[:3, :3])) for a, b in zip(pair[:-1], pair[1:])])
    waist.append(np.abs(np.diff(d['waist_target'], axis=0)).max(axis=1))
for name, values, unit, scale in (('left target step', move, 'mm', 1e3), ('left target turn', turn, 'deg', 180 / np.pi), ('pair step', pair_move, 'mm', 1e3),
                                  ('pair turn', pair_turn, 'deg', 180 / np.pi), ('waist target step', waist, 'deg', 180 / np.pi)):
    v = np.concatenate([np.asarray(x) for x in values]) * scale
    print(f'{name:18s} per sample [{unit}]: p50 {np.percentile(v,50):6.2f}  p95 {np.percentile(v,95):6.2f}  p99 {np.percentile(v,99):6.2f}  p99.9 {np.percentile(v,99.9):6.2f}  max {v.max():6.2f}   (x15 per second)')
