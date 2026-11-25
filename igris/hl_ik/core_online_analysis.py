"""Timing and tracking evidence from a completed or interrupted online trial."""
import argparse
import json
import re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .common import OUTPUT,save_json


def distribution(values):
    a=np.asarray(values,float)
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),
        p99=float(np.percentile(a,99)),max=float(a.max()),min=float(a.min()))


def analyze(name):
    out=OUTPUT/'native'/name
    result=json.loads((out/'result.json').read_text())
    commands=np.load(out/'commands.npz')['records'];mask=commands[:,2]==2;moving=commands[mask]
    if len(moving)<2:raise ValueError('trial has fewer than two moving samples')
    fixture=np.load(out/'targets.npz');updates=json.loads((out/'ik_updates.json').read_text())
    core=json.loads((out/'core_trace.json').read_text());trace=core['trace']
    start=max(i for i,v in enumerate(trace) if v['index']==0);trace=trace[start:]
    online=np.load(out/'online.npz')['records']
    elapsed=np.diff(moving[:,0]);ik=[json.loads(v) for v in (out/'ik.jsonl').read_text().splitlines()]
    checks=[json.loads(v) for v in (out/'guard.jsonl').read_text().splitlines()]
    received=[json.loads(v) for v in (out/'dds_received.jsonl').read_text().splitlines()]
    rates=[]
    for line in (out/'vendor.log').read_text().splitlines():
        match=re.search(r'\bsim: \[([^]]+)\].*\[LOW_LEVEL_WALK_MODE\] LowCmd: ([0-9.]+) Hz',line)
        if match:
            stamp=datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc).timestamp()
            if np.ceil(moving[0,0])<stamp<np.floor(moving[-1,0]):rates.append(float(match[2]))
    receipt={}
    if received:
        received=np.asarray(received);indices=received[:,2].astype(int)-1
        valid=(indices>=0)&(indices<len(commands));received=received[valid];indices=indices[valid]
        moving_mask=commands[indices,2]==2;received=received[moving_mask];indices=indices[moving_mask]
        receipt=dict(count=len(received),rate_hz=float((len(received)-1)/np.ptp(received[:,0])),
            interval_ms=distribution(np.diff(received[:,1])*1000),
            missing_sequences=int(np.maximum(0,np.diff(received[:,2])-1).sum()),
            q_matches_sent=bool(np.allclose(received[:,3:],commands[indices][:,3+np.r_[0:3,15:29]],atol=2e-7,rtol=0)),
            receive_minus_postpublish_ms=distribution((received[:,0]-commands[indices,0])*1000))
    m=dict(result=result,publisher=dict(count=len(moving),rate_hz=float((len(moving)-1)/np.ptp(moving[:,0])),
        interval_ms=distribution(elapsed*1000),interval_over_15ms=int((elapsed>.015).sum()),interval_over_20ms=int((elapsed>.02).sum()),
        validation_ms=distribution([v['validation_s']*1000 for v in trace]),
        lateness_ms=distribution([v['lateness_s']*1000 for v in trace]),
        skipped_samples=core['status']['skipped_samples']),native_dds=receipt,
        controller_lowcmd_rate_hz=distribution(rates),
        ik=dict(count=len(ik),solve_ms=distribution([v['solve_ms'] for v in ik if 'solve_ms' in v]),
            qp_ms=distribution([v['qp_ms'] for v in ik if 'qp_ms' in v]),
            collision_ms=distribution([v['collision_ms'] for v in ik if 'collision_ms' in v]),
            service_ms=distribution([(v['end']-v['begin'])*1000 for v in ik]),
            over_10ms=sum(v['solve_ms']>10 for v in ik if 'solve_ms' in v),
            over_33_33ms=sum(v['end']-v['begin']>1/30 for v in ik),
            skipped_goal_frames=int(np.maximum(0,np.diff([v['frame'] for v in ik if 'frame' in v])-1).sum()),
            position_mm=distribution([v['position_m']*1000 for v in ik if 'position_m' in v]),
            orientation_deg=distribution([np.rad2deg(v['orientation_rad']) for v in ik if 'orientation_rad' in v]),
            rejected=sum(not v['accepted'] for v in ik),
            result_application_age_ms=distribution([(v['applied_progress']-v['goal_time'])*1000 for v in updates])),
        collision=dict(checks=len(checks),rejected=sum(not v['accepted'] for v in checks),
            service_ms=distribution([(v['end']-v['begin'])*1000 for v in checks])),
        interpolation=dict(horizon_s=json.loads((out/'trial.json').read_text())['horizon_s'],
            sample_compute_ms=distribution(online[:,3]*1000),
            commanded_velocity_max_rad_s=float(abs(online[:,35:66][:,np.r_[0:3,15:29]]).max()),
            commanded_acceleration_max_rad_s2=float(abs(online[:,66:97][:,np.r_[0:3,15:29]]).max()),
            commanded_jerk_max_rad_s3=float(abs(online[:,97:128][:,np.r_[0:3,15:29]]).max()),
            q_step_max_deg=float(np.rad2deg(abs(np.diff(moving[:,3+np.r_[0:3,15:29]],axis=0))).max())),
        definitions=dict(ik_history='own nominal-30Hz IK solutions; no measured-posture neural feedback',
            native_dds='independent ROS LowCmd subscriber over CycloneDDS, matching bridge_sim input; timestamps include reader scheduling. Native controller reception is checked separately using its rate log.',
            collision='every outgoing full measured-lower-body posture plus sampled 0.005rad corridor; configured hand envelope',
            end_to_end='tracking uses contemporaneous requested goal, retaining IK and interpolation delay',
            derivative_clock='analytic core curve derivatives with respect to native simulation time'),hardware_executed=False)
    if 'validation_started_at' in trace[0]:
        worst=max(trace,key=lambda r:r['lateness_s'])
        m['worst_deadline']=worst
    gc_spans=[]
    if (out/'gc_events.json').exists():
        starts={}
        for stamp,phase,generation in json.loads((out/'gc_events.json').read_text()):
            if phase=='start':starts[generation]=stamp
            elif generation in starts:gc_spans.append(dict(start=starts.pop(generation),end=stamp,generation=generation))
        m['publisher_gc_spans']=gc_spans
    for kind,rows in [('ik',ik),('collision',checks)]:
        m[kind]['generation2_gc_calls']=[r for r in rows if any(e[2]==2 for e in r.get('gc_events',[]))]
    save_json(out/'online_metrics.json',m)
    # These are post-run diagnostic artifacts, never input to the online run.
    plan=OUTPUT/'box_plans'/name;plan.mkdir(exist_ok=True)
    actual=commands[commands[:,2]>=2];t,idx=np.unique(actual[:,1],return_index=True);q=actual[idx,3:]
    np.savez_compressed(plan/'motion.npz',time=t,q=q)
    np.savez_compressed(plan/'ik.npz',time=fixture['time'],goals=fixture['goals'],
        q=np.column_stack([np.interp(fixture['time'],t,q[:,i]) for i in range(31)]))
    save_json(plan/'provenance.json',dict(source=str(out/'commands.npz'),generated_after_execution=True))
    print(json.dumps(m,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();analyze(a.name)
