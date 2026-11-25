"""Results table over batches: one row per condition (a batch folder, optionally a tag part and a set of episodes).
usage: table.py name=folder[:tagpart][@7,13,...] ...      (batch folders under $UMI_OUTPUTS, default outputs/umi_act)"""
import json, os, sys
from pathlib import Path
import numpy as np

def rows_of(spec):
    name, rest = spec.split('=')
    episodes = None
    if '@' in rest:
        rest, eps = rest.split('@'); episodes = {int(e) for e in eps.split(',')}
    folder, _, part = rest.partition(':')
    rows = [json.loads(line) for line in (Path(os.environ.get('UMI_OUTPUTS', 'outputs/umi_act')) / folder / 'summary.jsonl').read_text().splitlines()]
    rows = [r for r in rows if 'stacked' in r and (not part or f'_{part}_' in r['run']) and (episodes is None or r['episode'] in episodes)]
    return name, rows

print(f"{'condition':28s} runs  3 placed  3 stacked   placed  stacked   lifted | pelvis cm / deg | IK p99 mm  accel p99  >5Hz   limit  failed  skipped")
for spec in sys.argv[1:]:
    name, rows = rows_of(spec)
    if not rows:
        print(f'{name:28s} no runs'); continue
    n = len(rows); med = lambda key: float(np.median([r[key] for r in rows]))
    print(f"{name:28s} {n:4d}  {sum(r['all_placed'] for r in rows):3d}/{n:<3d}  {sum(r['all_stacked'] for r in rows):4d}/{n:<3d}  "
          f"{sum(r['placed'] for r in rows):3d}/{3*n:<3d} {sum(max(r['stacked'],0) for r in rows):3d}/{3*n:<3d}  {sum(r['lifted'] for r in rows):3d}/{3*n:<3d} | "
          f"{med('pelvis_move_cm'):5.1f} / {med('pelvis_turn_deg'):5.1f}   | {med('ik_err_p99_mm'):7.1f}  {med('accel_p99'):9.1f}  {med('above_5hz'):.3f}  {med('saturated'):.3f}  "
          f"{sum(not r['worker_ok'] for r in rows):3d}    {sum(r.get('skipped', 0) for r in rows):4d}"
          # Boxes in the order they are carried (the log lists them bottom box first; the top box is taken first).
          f"   | placed 1st/2nd/3rd: {'/'.join(str(sum(r['levels'][2 - order] >= 0 for r in rows)) for order in range(3))}")
