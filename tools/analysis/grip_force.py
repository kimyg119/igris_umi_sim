"""Normal force of the hands on a parcel while it is in the air (from contacts.jsonl of a run)."""
import json, sys, collections
import numpy as np
for spec in sys.argv[1:]:
    name, path = spec.split('=')
    result = json.load(open(path + '/run/replay_result.json')); start = result['replay_start_sim'] if 'replay_start_sim' in result else 0.
    log = np.load(path + '/run/replay_log.npz'); m = log['phase'] == 'replay'; sim = log['sim'][m]; z = log['parcels'][m][:, :, 2]
    force = collections.defaultdict(lambda: collections.defaultdict(float))       # parcel -> time bin -> summed normal force of hand geoms
    geoms = collections.Counter()
    for line in open(path + '/run/contacts.jsonl'):
        row = json.loads(line)
        if not row['room_body'].startswith('room_Parcel'): continue
        geoms[(row['robot_geom'], row['patch'])] += 1
        if 'hand' in row['robot_geom'] or row['patch'] != 'other':
            force[row['room_body']][round(row['sim'], 1)] += row['normal_force_N']
    print(name, 'hand geoms seen on parcels:', dict(geoms.most_common(6)))
    for index in (2, 1, 0):                                          # carried first, second, third
        air = (z[:, index] - z[0, index]) > .03
        if not air.any(): print(f'   parcel {index}: never in the air'); continue
        t0, t1 = sim[air][0], sim[air][-1]
        f = np.array([v for t, v in sorted(force[f'room_Parcel_{index}'].items()) if t0 <= t <= t1])
        if len(f): print(f'   parcel {index} in the air {t0-start:5.1f}-{t1-start:5.1f} s: summed hand normal force per record p10 {np.percentile(f,10):5.1f} N  median {np.median(f):5.1f} N  p90 {np.percentile(f,90):5.1f} N  (records {len(f)})')
