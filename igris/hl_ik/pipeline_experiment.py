"""Paired solver/prior ablations, full collision audits and command export."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from .common import OUTPUT,ROOT,HERE,SDK_NAMES,sha256,save_json,Rotation
from .solver import HLIK,ElbowPrior
from .pipeline import PipelineIK
from .box_plan import box_targets

FULL=OUTPUT/'training/full/seed_42/best.pt'
REFINED=ROOT/'outputs/hl_ik_refined_training_20260915/snapshot_v1/training/refined_retimed/seed_42/best.pt'
CONFIGS={
    'lm_none':dict(backend='lm',prior=None),
    'lm_full':dict(backend='lm',prior=FULL),
    'lm_refined':dict(backend='lm',prior=REFINED),
    'qp_weighted_full':dict(backend='weighted',prior=FULL,acceleration_limit=None),
    'qp_priority_unbounded':dict(backend='priority',prior=FULL,acceleration_limit=None),
    'qp_priority_none':dict(backend='priority',prior=None,acceleration_limit=30.),
    'qp_priority_full':dict(backend='priority',prior=FULL,acceleration_limit=30.),
    'qp_priority_refined':dict(backend='priority',prior=REFINED,acceleration_limit=30.),
    'qp_priority_precision':dict(backend='priority',prior=FULL,acceleration_limit=6.),
    'qp_priority_weak_full':dict(backend='priority',prior=FULL,acceleration_limit=30.,prior_scale=.1),
    'qp_priority_weak_refined':dict(backend='priority',prior=REFINED,acceleration_limit=30.,prior_scale=.1),
}

def make_solver(label,box=False):
    cfg=CONFIGS[label];prior=ElbowPrior(cfg['prior']) if cfg['prior'] else None
    args=dict(hand_envelope=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/compliance.json') if box else {}
    return HLIK(prior,**args) if cfg['backend']=='lm' else PipelineIK(prior,mode=cfg['backend'],acceleration_limit=cfg['acceleration_limit'],prior_scale=cfg.get('prior_scale',1.),**args)

def prepare(out):
    (out/'fixtures').mkdir(parents=True,exist_ok=False)
    ts,goals,reference,_=box_targets()
    np.savez_compressed(out/'fixtures/box.npz',time=ts,goals=goals,reference=reference,warmup=reference[:1])
    records=[dict(name='box',kind='box',file='box.npz')]
    fixtures=json.loads((OUTPUT/'evaluation/rollout_fixtures.json').read_text())
    for clip in fixtures['clips']:
        with np.load(clip['output']) as z:
            begin=clip['start'];names=z['joint_names'].tolist();source=z['qpos'][begin:begin+125]
            qs=np.zeros((125,31))
            for j,name in enumerate(SDK_NAMES[:29]):qs[:,j]=source[:,7+names.index(name)]
            states=z['state'][begin:begin+125];goals=states[5:,:,:7].copy()
            goals[:,:,:3]+=np.array([[0,.171,.266],[0,-.171,.266]])
        name=clip['path'].split('/')[0]
        np.savez_compressed(out/'fixtures'/(name+'.npz'),time=np.arange(120)/30,goals=goals,reference=qs[5:],warmup=qs[:5],elbows=states[5:,:,7:])
        records.append(dict(name=name,kind='recorded',file=name+'.npz',source=clip['path'],source_sha256=sha256(clip['output']),group=clip['group']))
    config={k:{n:str(v) if isinstance(v,Path) else v for n,v in d.items()} for k,d in CONFIGS.items()}
    save_json(out/'manifest.json',dict(fixtures=records,configs=config,fixture_sha256={r['name']:sha256(out/'fixtures'/r['file']) for r in records},
        prior_sha256={str(p):sha256(p) for p in [FULL,REFINED]},
        protocol='same targets, warmup, lower-body reference, full sphere model and 0.01rad corridor; stop at first non-accepted tick; score remaining frames as held failures',
        history='observed 30Hz initialization then own accepted FK; recorded initial velocity from last two observations; box starts stationary',hardware_executed=False))

def run(out,label):
    manifest=json.loads((out/'manifest.json').read_text());dest=out/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
    rows=[]
    for fixture in manifest['fixtures']:
        z=np.load(out/'fixtures'/fixture['file']);times=z['time'];goals=z['goals'];reference=z['reference'];warmup=z['warmup']
        solver=make_solver(label,fixture['kind']=='box');q=warmup[-1].copy();solver.reset(warmup[0])
        for obs in warmup[1:]:solver.observe(obs)
        if isinstance(solver,PipelineIK) and len(warmup)>1:
            solver.velocity=(warmup[-1,15:29]-warmup[-2,15:29])*30
        commands=[];metrics=[];reasons=[];predictions=[];stop=None
        tolerance=.015 if fixture['kind']=='box' else .005
        for i,goal in enumerate(goals):
            previous=q.copy();elapsed=solve_ms=qp_ms=collision_ms=0.;qp_fallbacks=0;accepted=False;prior_status='not_called'
            if stop is None:
                input_q=q.copy();input_q[:15]=reference[i,:15]
                start=time.perf_counter();r=solver.solve(goal,input_q,position_tolerance=tolerance,record_history=False)
                accepted=r.accepted if isinstance(solver,PipelineIK) else r.reason!='collision'
                solve_ms=r.elapsed_ms;qp_ms=getattr(r,'qp_ms',0.);collision_ms=getattr(r,'collision_ms',0.)
                qp_fallbacks=getattr(r,'qp_fallbacks',0)
                prior_status=getattr(r,'prior_status','used' if solver.prior else 'disabled');reason=r.reason
                if accepted:
                    # Include caller-owned lower-body motion in the final gate.
                    solver.set_q(previous);a=solver.configuration.q.copy();solver.set_q(r.q);b=solver.configuration.q.copy()
                    hit=solver.geometry.corridor(solver.configuration.data,a,b,max_step=.01)
                    if hit:accepted=False;reason='caller_corridor_collision'
                if accepted:q=r.q;solver.observe(q)
                else:stop=dict(frame=i,time_s=float(times[i]),reason=reason)
                elapsed=(time.perf_counter()-start)*1000
                prediction=r.predicted_elbow
            else:reason='held_after_stop';prediction=None
            solver.set_q(q);states=solver.states();actual=states[:,:7].copy();actual[:,:3]+=solver.shoulder
            pe=np.linalg.norm(actual[:,:3]-goal[:,:3],axis=1).max()
            ae=(Rotation.from_quat(actual[:,3:],scalar_first=True).inv()*Rotation.from_quat(goal[:,3:],scalar_first=True)).magnitude().max()
            elbow=np.linalg.norm(states[:,7:10]-z['elbows'][i,:,:3],axis=1).mean() if 'elbows' in z else np.nan
            success=accepted and pe<=tolerance and ae<=np.deg2rad(2)
            metrics.append([pe,ae,solve_ms,elapsed,success,accepted,qp_ms,collision_ms,elbow,qp_fallbacks])
            commands.append(q.copy());reasons.append(reason);predictions.append(np.full((2,7),np.nan) if prediction is None else prediction)
            if i%600==0:print(label,fixture['name'],i,len(goals),'mm',round(pe*1000,3),reason,flush=True)
        commands=np.array(commands);a=np.array(metrics);called=a[:,3]>0;v=np.diff(np.vstack((warmup[-1:],commands))[:,15:29],axis=0)*30
        initial_velocity=np.zeros((1,14)) if len(warmup)==1 else (warmup[-1:,15:29]-warmup[-2:-1,15:29])*30
        acc=np.diff(np.vstack((initial_velocity,v)),axis=0)*30
        row=dict(name=fixture['name'],frames=len(times),completed=stop is None,stop=stop,accepted_percent=float(a[:,5].mean()*100),
            strict_pass_percent=float(a[:,4].mean()*100),position_mean_mm=float(a[:,0].mean()*1000),position_max_mm=float(a[:,0].max()*1000),
            orientation_mean_deg=float(np.rad2deg(a[:,1]).mean()),elbow_mean_mm=float(np.nanmean(a[:,8])*1000) if 'elbows' in z else None,
            solver_ms={k:float(x) for k,x in zip(['median','p95','p99','max'],np.percentile(a[called,2],[50,95,99,100]))},
            total_ms={k:float(x) for k,x in zip(['median','p95','p99','max'],np.percentile(a[called,3],[50,95,99,100]))},
            calls=int(called.sum()),deadline_miss_percent=float((a[called,3]>1000/30).mean()*100),
            qp_fallbacks=int(a[:,9].sum()),
            velocity_max=float(abs(v).max()),acceleration_max=float(abs(acc).max()),jerk_p95=float(np.percentile(abs(np.diff(acc,axis=0)*30),95)),
            collision=solver.geometry.provenance())
        np.savez_compressed(dest/(fixture['name']+'.npz'),time=times,q=commands,goals=goals,metrics=a,reasons=reasons,predicted_elbow=predictions)
        rows.append(row);save_json(dest/'metrics.json',dict(label=label,fixtures=rows,manifest_sha256=sha256(out/'manifest.json'),
            source_sha256={n:sha256(HERE/n) for n in ['pipeline.py','pipeline_experiment.py','solver.py','geometry.py','retarget_quality.py']},
            columns=['position_m','orientation_rad','solve_ms','total_ms','strict_pass','accepted','qp_ms','collision_ms','elbow_m','qp_fallbacks']))
        print(json.dumps(dict(label=label,**{k:v for k,v in row.items() if k!='collision'})),flush=True)

def export(out,label,name):
    """Offline C2 interpolation with dense pose/limit/collision revalidation."""
    source=out/'runs'/label;meta=json.loads((source/'metrics.json').read_text());box=next(r for r in meta['fixtures'] if r['name']=='box')
    if not box['completed']:raise ValueError('cannot export a stopped box path')
    z=np.load(source/'box.npz');t=z['time'];q=z['q']
    spline=CubicSpline(t,q,axis=0,bc_type=((1,np.zeros(31)),(1,np.zeros(31))))
    dense=np.unique(np.r_[np.linspace(t[0],t[-1],round(t[-1]*1000)+1),t])
    raw_v=float(abs(spline(dense,1)[:,15:29]).max());raw_a=float(abs(spline(dense,2)[:,15:29]).max())
    scale=max(1.,raw_v/3.,np.sqrt(raw_a/30.))
    if scale>1.:scale*=1.001
    duration=np.ceil(t[-1]*scale*100)/100;scale=duration/t[-1]
    time100=np.arange(round(duration*100)+1)/100;source100=np.minimum(time100/scale,t[-1])
    command=spline(source100);solver=make_solver(label,box=True);previous=None;errors=[];collisions=0;limit_failures=0
    from scipy.spatial.transform import Slerp
    goals=np.empty((len(time100),2,7))
    for side in range(2):
        for j in range(3):goals[:,side,j]=np.interp(source100,t,z['goals'][:,side,j])
        goals[:,side,3:]=Slerp(t,Rotation.from_quat(z['goals'][:,side,3:],scalar_first=True))(source100).as_quat(scalar_first=True)
    for cur,goal in zip(command,goals):
        solver.set_q(cur);b=solver.configuration.q.copy();arm=cur[15:29]
        if np.any(arm<solver.limits[:,0]-1e-7) or np.any(arm>solver.limits[:,1]+1e-7) or np.any(solver.A@arm>solver.b+1e-7):limit_failures+=1
        hit=solver.geometry.corridor(solver.configuration.data,b if previous is None else previous,b,max_step=.005)
        collisions+=hit is not None;previous=b
        actual=solver.states()[:,:7];actual[:,:3]+=solver.shoulder
        errors.append([np.linalg.norm(actual[:,:3]-goal[:,:3],axis=1).max(),
            (Rotation.from_quat(actual[:,3:],scalar_first=True).inv()*Rotation.from_quat(goal[:,3:],scalar_first=True)).magnitude().max()])
    e=np.array(errors);velocity=float(raw_v/scale);acceleration=float(raw_a/scale**2)
    validation=dict(collisions=collisions,limit_failures=limit_failures,velocity_max=velocity,acceleration_max=acceleration,
        strict_pass_percent=float(((e[:,0]<=.015)&(e[:,1]<=np.deg2rad(2))).mean()*100),position_mean_mm=float(e[:,0].mean()*1000),
        position_max_mm=float(e[:,0].max()*1000),time_scale=float(scale),original_duration_s=float(t[-1]),duration_s=float(duration),
        interpolation='C2 cubic with zero endpoint velocity; explicit time scaling to3rad/s and30rad/s2; derivatives audited at1kHz; full spheres and0.005rad corridors at100Hz')
    valid=bool(not collisions and not limit_failures and velocity<=3.00001 and acceleration<=30.0001)
    validation['simulation_diagnostic_export']=valid
    dest=OUTPUT/'box_plans'/name;dest.mkdir(parents=True,exist_ok=False);save_json(dest/'command_validation.json',validation)
    if not valid:raise ValueError('command interpolation failed validation')
    np.savez_compressed(dest/'motion.npz',time=time100,q=command,source_time=source100,time_scale=scale)
    np.savez_compressed(dest/'ik.npz',time=t*scale,q=q,goals=z['goals'],statistics=z['metrics'][:,[0,1,2,5,4]])
    save_json(dest/'metrics.json',dict(name=name,source=str(source),source_sha256=sha256(source/'box.npz'),completed=True,
        strict_tracking_pass_fraction=box['strict_pass_percent']/100,position_mean_m=box['position_mean_mm']/1000,
        status='offline simulation diagnostic; not hardware authority',command_validation=validation))
    print(json.dumps(validation),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','export']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--label',choices=list(CONFIGS));p.add_argument('--name');a=p.parse_args()
    if a.command=='prepare':prepare(a.out)
    elif a.command=='run':run(a.out,a.label)
    else:export(a.out,a.label,a.name)
