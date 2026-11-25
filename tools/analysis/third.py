"""What became of the third box (the lowest one, carried last) in each run of a batch."""
import json, sys
from pathlib import Path
import numpy as np
from igris.umi_walklow_replay.analyze import tilt_deg, yaw
for folder in sorted(Path(sys.argv[1]).glob('ep*')):
    run = folder / 'run'
    if not (run / 'replay_log.npz').exists(): continue
    result = json.loads((run / 'replay_result.json').read_text()); manifest = json.loads((run / 'manifest.json').read_text())['umi_replay']
    log = np.load(run / 'replay_log.npz'); m = np.isin(log['phase'], ('replay', 'hold'))
    sim = log['sim'][m] - log['sim'][log['phase'] == 'replay'][0]; box = log['parcels'][m][:, 0]
    layout = manifest['layout']; staged = np.array(result['pelvis_at_staging']); h = float(yaw(staged[3:])); c, s = np.cos(h), np.sin(h)
    place = np.array(layout['place']['xy']); place = staged[:2] + np.array([c * place[0] - s * place[1], s * place[0] + c * place[1]])
    level = layout['surface_top_m'] + 2.5 * layout['parcel_size_m'][2]
    rise = box[:, 2] - box[0, 2]; up = np.flatnonzero(rise > .03)
    first_up = sim[up[0]] if len(up) else None
    pelvis = log['pelvis'][m]; turn = np.degrees(np.unwrap(yaw(pelvis[:, 3:]))); turn -= turn[0]
    fate = 'never lifted' if first_up is None else 'placed' if abs(box[-1, 2] - level) < .03 and np.linalg.norm(box[-1, :2] - place) < .1 and tilt_deg(box[-1, 3:]) < 15 \
        else 'on the floor' if box[-1, 2] < .3 else 'in the air at the end' if abs(np.diff(box[-30:, 2]).sum()) > .005 or (box[-1, 2] > level + .05) else f'elsewhere (z {box[-1,2]:.2f}, {np.linalg.norm(box[-1,:2]-place)*100:.0f} cm from its place, tilt {tilt_deg(box[-1,3:]):.0f})'
    i = up[0] if len(up) else len(sim) - 1
    print(f"{folder.name:24s} third box: {fate:55s} lifted at {'-' if first_up is None else round(float(first_up),1)} s of {sim[-1]:.0f} s; pelvis turn then {turn[i]:6.1f} deg, at the end {turn[-1]:6.1f} deg; max rise {rise.max()*100:.0f} cm")
