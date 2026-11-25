"""What happened in each run of a batch: boxes, pelvis, command quality, policy timing.  One JSON line per run and a table.

Two verdicts on the boxes.  ``stacked`` is analyze.py's, as in the earlier reports: each box upright at the stack level
of its turn (top box first) at the end of the log.  ``placed`` does not use the order or the timing of the recording (a
policy keeps its own): a box counts when at the end it stands upright at one of the three stack levels within 10 cm of
the recorded set-down point.  ``lifted``: it ever rose 3 cm above where it stood."""
import json, sys
from pathlib import Path
import numpy as np

from igris.umi_walklow_replay.analyze import tilt_deg, yaw
from igris.umi_walklow_replay.kinematics import ACTIVE


def summary(out):
    out = Path(out); run = out / 'run'
    result = json.loads((run / 'replay_result.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())['umi_replay']
    log = np.load(run / 'replay_log.npz')
    phase = log['phase']; moving = phase == 'replay'
    row = dict(run=out.name, episode=manifest['episode'], worker_ok=bool(result.get('pass')), error=(result.get('error') or '')[:60])
    if not moving.any():
        return row
    layout = manifest['layout']; staged = np.array(result['pelvis_at_staging'])
    heading = float(yaw(staged[3:])); c, s = np.cos(heading), np.sin(heading)
    place = np.array(layout['place']['xy']); place = staged[:2] + np.array([c * place[0] - s * place[1], s * place[0] + c * place[1]])
    levels = layout['surface_top_m'] + (np.arange(3) + .5) * layout['parcel_size_m'][2]
    after = np.concatenate([log['parcels'][moving], log['parcels'][phase == 'hold']])
    start = log['parcels'][moving][0]; final = after[-1]
    lifted = (after[:, :, 2].max(axis=0) - start[:, 2]) >= .03
    at_level = [int(np.argmin(np.abs(levels - box[2]))) if np.abs(levels - box[2]).min() <= .03 and tilt_deg(box[3:]) <= 15.
                and np.linalg.norm(box[:2] - place) <= .10 else -1 for box in final]
    placed = len({level for level in at_level if level >= 0})
    analysis = json.loads((run / 'analysis.json').read_text()) if (run / 'analysis.json').exists() else {}
    sim = log['sim'][moving]; q = log['q_command'][moving][:, ACTIVE]
    dt = np.maximum(np.diff(sim), 1e-4)
    speed = np.diff(q, axis=0) / dt[:, None]; accel = np.abs(np.diff(speed, axis=0)) / dt[1:, None]
    # Share of the joint velocity's energy above 5 Hz (the commands resampled to 100 Hz).
    grid = np.arange(sim[0], sim[-1], .01); even = np.array([np.interp(grid, sim, q[:, j]) for j in range(q.shape[1])]).T
    spectrum = np.abs(np.fft.rfft(np.diff(even, axis=0) * 100., axis=0)) ** 2; freq = np.fft.rfftfreq(len(even) - 1, .01)
    pelvis = log['pelvis'][moving]
    turn = np.degrees(np.unwrap(yaw(pelvis[:, 3:])))
    row.update(
        lifted=int(lifted.sum()), stacked=int(analysis.get('boxes_stacked', -1)), all_stacked=bool(analysis.get('stack_success', False)),
        placed=int(placed), all_placed=bool(placed == 3), levels=at_level,
        seconds=round(float(sim[-1] - sim[0]), 1),
        ik_err_p50_mm=round(1e3 * float(np.percentile(log['position_error'][moving], 50)), 1),
        ik_err_p99_mm=round(1e3 * float(np.percentile(log['position_error'][moving], 99)), 1),
        speed_max=round(float(np.abs(speed).max()), 2), accel_p99=round(float(np.percentile(accel.max(axis=1), 99)), 1),
        above_5hz=round(float(spectrum[freq > 5.].sum() / max(spectrum[freq > .05].sum(), 1e-12)), 4),
        saturated=round(float(log['saturated'][moving].mean()), 3), relaxed=int(np.count_nonzero(log['relaxed'][moving])),
        pelvis_move_cm=round(100 * float(np.linalg.norm(pelvis[:, :2] - pelvis[0, :2], axis=1).max()), 1),
        pelvis_turn_deg=round(float(np.abs(turn - turn[0]).max()), 1), pelvis_min_z=round(float(pelvis[:, 2].min()), 3),
        anchor=result.get('anchor'), waist=(result.get('policy') or {}).get('waist'))
    policy = result.get('policy')
    if policy:
        row.update(chunks=policy['chunks'], skipped=policy['skipped_samples'], arrival_max=policy['arrival_samples_after_observation_max'])
    return row


if __name__ == '__main__':
    rows = []
    for folder in sorted(Path(sys.argv[1]).glob('ep*')):
        if not (folder / 'run/replay_log.npz').exists():
            continue
        try:
            rows.append(summary(folder))
        except Exception as error:
            rows.append(dict(run=folder.name, failed=repr(error)[:160]))
        print(json.dumps(rows[-1]), flush=True)
    good = [row for row in rows if 'stacked' in row]
    if good:
        print('SUMMARY', json.dumps(dict(runs=len(good), all_stacked=sum(row['all_stacked'] for row in good),
                                         all_placed=sum(row['all_placed'] for row in good),
                                         boxes_stacked=sum(max(row['stacked'], 0) for row in good), boxes_placed=sum(row['placed'] for row in good),
                                         boxes_lifted=sum(row['lifted'] for row in good),
                                         boxes=3 * len(good), worker_failed=sum(not row['worker_ok'] for row in good))))
    (Path(sys.argv[1]) / 'summary.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
