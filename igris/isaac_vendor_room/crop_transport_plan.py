"""Start an already validated HL-IK plan before contact, then re-audit retiming.

The selected starting posture is reached by the private simulator's preparation
ramp before room initialization. No object state is changed during the task.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from transport_plan import PipelineIK,ENVELOPE,sha256,load_motion


def crop(source,out,model,start=20.,acceleration=1.):
    report=json.loads((source/'plan_result.json').read_text())
    if not report.get('exported'):raise ValueError('Exported validated source required')
    out.mkdir(parents=True,exist_ok=False)
    targets=np.load(source/'targets.npz');ik=np.load(source/'ik.npz');old=np.load(source/'motion_100hz.npz')
    first=int(np.searchsorted(ik['time'],start));offset=float(ik['time'][first]);t=ik['time'][first:]-offset;q=ik['q'][first:]
    if len(t)<4:raise ValueError('Insufficient remaining motion')
    original_factor=report['time_scale'];spline=CubicSpline(t,q,axis=0,bc_type=((1,np.zeros(31)),(1,np.zeros(31))))
    owned=old['command_sdk_indices'];dense=np.linspace(0,t[-1],round(t[-1]*1000)+1)
    bounds=[float(np.max(abs(spline(dense,n)[:,owned]))) for n in (1,2,3)]
    scale=max(1,bounds[0]/.49,np.sqrt(bounds[1]/acceleration),np.cbrt(bounds[2]/30))*1.002
    duration=np.ceil(t[-1]*scale*100)/100;scale=duration/t[-1];stamps=np.arange(round(duration*100)+1)/100
    sample_time=stamps/scale;command=spline(sample_time)
    goal=targets['targets'][first:];rotations=[Slerp(t,Rotation.from_quat(goal[:,side,3:],scalar_first=True))(sample_time).as_matrix() for side in range(2)]
    positions=np.stack([np.array([np.interp(sample_time,t,goal[:,s,j]) for j in range(3)]).T for s in range(2)],axis=1)
    solver=PipelineIK(model_path=model,hand_envelope=ENVELOPE);last=None;max_error=np.zeros(2)
    for i,cur in enumerate(command):
        solver.set_q(cur);configuration=solver.configuration.q.copy();arm=cur[15:29]
        limited=solver.model.jnt_limited.astype(bool);v=configuration[solver.model.jnt_qposadr[limited]]
        if np.any(v<solver.model.jnt_range[limited,0]-1e-8) or np.any(v>solver.model.jnt_range[limited,1]+1e-8) or np.any(solver.A@arm>solver.b+1e-8):raise ValueError(f'Interpolated limits {i}')
        hit=solver.geometry.corridor(solver.configuration.data,configuration if last is None else last,configuration,max_step=.005)
        if hit:raise ValueError(f'Interpolated collision {i}: {hit}')
        for side,task in enumerate(solver.tasks[:2]):
            actual=solver.configuration.get_transform_frame_to_world(task.frame_name,'body').as_matrix()
            max_error=np.maximum(max_error,[np.linalg.norm(actual[:3,3]-positions[i,side]),np.rad2deg(Rotation.from_matrix(actual[:3,:3].T@rotations[side][i]).magnitude())])
        last=configuration
    if max_error[0]>.003 or max_error[1]>1:raise ValueError(f'Interpolated EE error {max_error}')
    np.savez_compressed(out/'targets.npz',**{k:(t if k=='time' else targets[k][first:]) for k in targets.files})
    np.savez_compressed(out/'ik.npz',**{k:(t if k=='time' else ik[k][first:]) for k in ik.files})
    old_time=(sample_time+offset)*original_factor
    np.savez_compressed(out/'motion_100hz.npz',command_t_s=stamps,q_command_rad=command[:,owned],command_sdk_indices=owned,
        mode=np.full(len(stamps),'FOLLOWING'),trip=np.full(len(stamps),''),
        original_command_t_s=np.interp(old_time,old['command_t_s'],old['original_command_t_s']),
        source_time_s=np.interp(old_time,old['command_t_s'],old['source_time_s']),source_offset_s=old['source_offset_s'],
        frame='torso',rate_hz=100.,motion_knot_t_s=t*scale,motion_knot_q_rad=q[:,owned],
        expected_box_yaw_rad=np.interp(old_time,old['command_t_s'],old['expected_box_yaw_rad']),target_rotation=old['target_rotation'])
    load_motion(out/'motion_100hz.npz')
    report.update(source_plan=str(source),source_plan_sha256=sha256(source/'motion_100hz.npz'),preparation_through_plan_time_s=offset,
        phases={k:v-offset for k,v in report['phases'].items()},grasp_time=report['grasp_time']-offset,
        duration_s=duration,time_scale=scale,frames=len(t),requested_frames=len(t),
        derivative_bounds=[bounds[i]/scale**(i+1) for i in range(3)],motion_sha256=sha256(out/'motion_100hz.npz'),
        interpolated_fk_error=dict(position_m=float(max_error[0]),orientation_deg=float(max_error[1])),physics_steps=0)
    (out/'plan_result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','model'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start',type=float,default=20.);p.add_argument('--acceleration',type=float,default=1.)
    a=p.parse_args();crop(a.source.resolve(),a.out.resolve(),a.model.resolve(),a.start,a.acceleration)
