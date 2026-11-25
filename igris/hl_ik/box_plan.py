"""Closed-loop kinematic IK at 30 Hz, followed by 100 Hz native command interpolation.

Targets are the feasible EE FK from the preceding 1 kg experiment. Only the
initial arm q and the independently controlled waist/leg reference are reused.
Every subsequent elbow/history comes from this solver's own generated posture.
Residual failures remain visible; these files are offline simulation diagnostics.
"""
import argparse,time
import numpy as np
from scipy.spatial.transform import Slerp
from .common import *
from .solver import HLIK,ElbowPrior

def box_targets():
    """The same independent hand goals, initial q and caller-owned lower body."""
    original=ROOT/'outputs/synthetic_box_motion_20260914'
    with np.load(original/'motion_100hz.npz') as f:
        t100=f['command_t_s'];source=f['source_time_s'];source_pose=f['ee_pose_torso'];arm=f['q_command_rad']
    with np.load(original/'prepared.npz') as f:
        qref=np.column_stack([np.interp(source,f['times'],f['q'][:,j]) for j in range(31)])
    qref[:,np.r_[0:3,15:29]]=arm;qref[:,29:]=0 # The learned torso model has locked neck.
    ts=np.arange(round(t100[-1]*30)+1)/30
    goals=np.empty((len(ts),2,7))
    for s in range(2):
        goals[:,s,:3]=np.column_stack([np.interp(ts,t100,source_pose[:,s,j]) for j in range(3)])
        goals[:,s,3:]=Slerp(t100,Rotation.from_quat(source_pose[:,s,3:],scalar_first=True))(ts).as_quat(scalar_first=True)
    reference=np.column_stack([np.interp(ts,t100,qref[:,j]) for j in range(31)])
    return ts,goals,reference,t100

def plan(name,scope=None,prior_scale=1.,checkpoint_path=None,output=OUTPUT):
    out=output/'box_plans'/name;out.mkdir(parents=True,exist_ok=False)
    original=ROOT/'outputs/synthetic_box_motion_20260914'
    ts,goals,reference,t100=box_targets()
    checkpoint=checkpoint_path or (OUTPUT/'training'/scope/'seed_42/best.pt' if scope else None)
    envelope=IGRIS/'sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/compliance.json'
    solver=HLIK(ElbowPrior(checkpoint) if checkpoint else None,prior_scale=prior_scale,hand_envelope=envelope)
    q=reference[0].copy();solver.reset(q);commands=[];stats=[];predictions=[];reasons=[];stop=None
    for i,goal in enumerate(goals):
        q[:15]=reference[i,:15]
        r=solver.solve(goal,q,position_tolerance=.015,record_history=False)
        if r.reason=='collision':
            stop=dict(frame=i,time_s=float(ts[i]),reason='self-collision corridor; remaining motion withheld')
            print(json.dumps(stop),flush=True);break
        q=r.q;solver.observe(q)
        commands.append(q.copy());stats.append([r.position_error_m,r.orientation_error_rad,r.elapsed_ms,r.iterations,r.success])
        predictions.append(np.full((2,7),np.nan) if r.predicted_elbow is None else r.predicted_elbow);reasons.append(r.reason)
        if i%100==0:print(name,i,len(goals),r.position_error_m,r.orientation_error_rad,flush=True)
    qs=np.asarray(commands);stats=np.asarray(stats);ts=ts[:len(qs)];goals=goals[:len(qs)]
    if len(qs)<2:
        save_json(out/'metrics.json',dict(name=name,stop=stop,completed=False));return
    command_time=t100[t100<=ts[-1]]
    interp=np.column_stack([np.interp(command_time,ts,qs[:,j]) for j in range(31)])
    np.savez_compressed(out/'motion.npz',time=command_time,q=interp)
    np.savez_compressed(out/'ik.npz',time=ts,q=qs,goals=goals,statistics=stats,predicted_elbow=predictions,reasons=reasons)
    velocity=np.diff(qs[:,15:29],axis=0)*30;accel=np.diff(velocity,axis=0)*30
    save_json(out/'metrics.json',dict(name=name,scope=scope,prior_scale=prior_scale,checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        target_source=str(original/'motion_100hz.npz'),target_source_sha256=sha256(original/'motion_100hz.npz'),
        target_definition='previous corrected torso EE FK; same 1 kg task path; no reference elbow beyond initial posture',
        history_rate_hz=30,command_rate_hz=100,frames=len(ts),completed=stop is None,stop=stop,strict_tracking_pass_fraction=float(stats[:,4].mean()),
        first_tracking_failure_s=float(ts[np.flatnonzero(stats[:,4]==0)[0]]) if (stats[:,4]==0).any() else None,
        source_sha256={n:sha256(HERE/n) for n in ('box_plan.py','solver.py','network.py','geometry.py')},
        position_mean_m=float(stats[:,0].mean()),position_max_m=float(stats[:,0].max()),orientation_mean_deg=float(np.rad2deg(stats[:,1]).mean()),
        orientation_max_deg=float(np.rad2deg(stats[:,1]).max()),solver_ms_median=float(np.median(stats[:,2])),solver_ms_p95=float(np.percentile(stats[:,2],95)),
        joint_velocity_max_rad_s=float(abs(velocity).max()),joint_acceleration_max_rad_s2=float(abs(accel).max()),
        collision=solver.geometry.provenance(),neck='locked at zero',status='offline simulation diagnostic; residual failures are not execution approval'))
    print(out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--scope',choices=['paper','full']);ap.add_argument('--prior-scale',type=float,default=1.)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--output',type=Path,default=OUTPUT)
    a=ap.parse_args();plan(a.name,a.scope,a.prior_scale,a.checkpoint,a.output)
