"""Derive +10 cm carry and +8 cm reach from saved q; no robot transport."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize,LinearConstraint
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from hands import ROOT,SOURCE,URDF as TORSO_URDF
for p in [ROOT/'igris/real_client',ROOT/'igris/sim_local/vr_box_eval']:
    sys.path.insert(0,str(p))
from fit_motion_limits import ArmFK
from replay_q import joint_limit_constraints,check_joint_limits,OWNED
from igris_core.joint_trajectory import JointTrajectory
from igris_ik.robot_geometry import RobotGeometry
from compliant_hands import CompliantHands
from box_replay import MOTION,load_motion


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    source=np.load(MOTION);raw=np.load(MOTION.with_name('raw_targets.npz'))
    phases={}
    for i in np.r_[0,np.flatnonzero(raw['stage'][1:]!=raw['stage'][:-1])+1]:
        phases[str(raw['stage'][i])]=float(np.interp(raw['t'][i]-source['source_offset_s'],source['source_time_s'],source['command_t_s']))
    def smooth(x):
        u=np.clip(x,0,1);return u**3*(10-15*u+6*u*u)
    def height(t):
        return .10*smooth((t-phases['probe_lift'])/(phases['carry_left']-phases['probe_lift']))*(1-smooth((t-phases['lower'])/(phases['open']-phases['lower'])))
    urdf=URDF.load(str(TORSO_URDF),load_meshes=False,load_collision_meshes=False)
    tree=ET.parse(TORSO_URDF);assert tree.find(".//joint[@name='base_joint']/child").get('link')=='torso'
    axis_audit={}
    for side,label in [('l','left'),('r','right')]:
        m=urdf.joint_map[side+'_hand_dof_joint_mcp'];e=next(j for j in urdf.robot.joints if j.child=='end_effector_'+label)
        assert m.parent==e.parent
        offset=e.origin[:3,3]-m.origin[:3,3];axis=np.asarray(m.axis);distance=np.linalg.norm(np.cross(offset,axis))
        assert distance<1e-9
        axis_audit[side]={'distance_to_axis_m':float(distance),'offset_along_axis_m':float(offset@axis)}
    fk=[ArmFK(urdf,s) for s in (0,1)]
    times=source['motion_knot_t_s'];q=np.zeros((len(times),31));q[:,OWNED]=source['motion_knot_q_rad']
    original=JointTrajectory(times,q);modified=q.copy();low,high,a,b=joint_limit_constraints()
    for i,t in enumerate(times):
        dz=height(t)
        if dz==0:continue
        for s,forward in enumerate(fk):
            idx=forward.indices;seed=q[i,idx];target=forward(seed);target[:3,3]+=[.8*dz,0,dz]
            initial=seed+modified[i-1,idx]-q[i-1,idx] if i else seed
            def residual(x):
                pose=forward(x)
                return np.r_[(pose[:3,3]-target[:3,3])*30,Rotation.from_matrix(pose[:3,:3]@target[:3,:3].T).as_rotvec()*3,(x-seed)*.003]
            rows=np.any(a[:,idx]!=0,axis=1)
            margin=.01
            result=minimize(lambda x:float(residual(x)@residual(x)),np.clip(initial,low[idx]+margin,high[idx]-margin),method='SLSQP',bounds=list(zip(low[idx]+margin,high[idx]-margin)),constraints=[LinearConstraint(a[rows][:,idx],-np.inf,b[rows]-margin)],options={'ftol':1e-12,'maxiter':150})
            modified[i,idx]=result.x
        if i%50==0:print('FIT',i,len(times),flush=True)
    check_joint_limits(modified)
    curve=JointTrajectory(times,modified);curve.check_position_limits(low,high,OWNED)
    bounds=np.array([curve.derivative_bound(n,OWNED) for n in (1,2,3)])
    factor=max(1.,bounds[0]/.5,np.sqrt(bounds[1]/3.),np.cbrt(bounds[2]/100.))*1.005
    duration=np.ceil(curve.duration_s*factor*100)/100;factor=duration/curve.duration_s
    curve=curve.retimed(factor);t=np.arange(round(duration*100)+1)/100;progress=t/factor;frames=curve.sample(t);check_joint_limits(frames)
    print('BOUNDS',bounds.tolist(),'TIME_SCALE',factor,flush=True)
    reference=original.sample(progress);prepared=np.load(MOTION.with_name('prepared.npz'))
    source_time=np.interp(progress,source['command_t_s'],source['source_time_s'])
    full=np.column_stack([np.interp(source_time,prepared['times'],prepared['q'][:,j]) for j in range(31)]);full[:,OWNED]=frames[:,OWNED]
    hand=CompliantHands();hand.spheres=json.loads((SOURCE/'compliance.json').read_text())['spheres']
    errors=[];poses=[]
    try:
        geometry=RobotGeometry();previous=full[0]
        for i,configuration in enumerate(full):
            reason=geometry.command_guard(previous,configuration)
            if reason:raise ValueError(f'Compliant-hand collision corridor {i}: {reason}')
            row=[];pose=[]
            for forward in fk:
                target=forward(reference[i,forward.indices]);dz=height(progress[i]);target[:3,3]+=[.8*dz,0,dz];actual=forward(frames[i,forward.indices])
                pe=np.linalg.norm(actual[:3,3]-target[:3,3]);ae=Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()
                if pe>.002 or ae>np.deg2rad(2.):raise ValueError(f'FK mismatch {i}: {pe}, {ae}')
                row.append([pe,ae]);pose.append(np.r_[actual[:3,3],Rotation.from_matrix(actual[:3,:3]).as_quat()[[3,0,1,2]]])
            errors.append(row);poses.append(pose);previous=configuration
            if i%1000==0:print('CHECK',i,len(t),flush=True)
    finally:hand.close()
    errors=np.array(errors)
    np.savez_compressed(out/'motion_100hz.npz',motion_knot_t_s=curve.times,motion_knot_q_rad=curve.positions[:,OWNED],command_t_s=t,q_command_rad=frames[:,OWNED],command_sdk_indices=OWNED,mode=np.full(len(t),'FOLLOWING'),trip=np.full(len(t),''),source_time_s=source_time,source_offset_s=source['source_offset_s'],ee_pose_torso=poses,fk_errors=errors,frame='torso',rate_hz=100.,original_command_t_s=progress,additional_z_m=height(progress))
    load_motion(out/'motion_100hz.npz')
    report=dict(pass_=True,source=str(MOTION),source_sha256=hashlib.sha256(MOTION.read_bytes()).hexdigest(),urdf=str(TORSO_URDF),urdf_sha256=hashlib.sha256(TORSO_URDF.read_bytes()).hexdigest(),ee_mcp_axis=axis_audit,frame='torso',additional_z_m=.10,additional_forward_m=.08,forward_reason='Preserve both hand orientations within existing wrist yaw and coupled limits',pickup_and_release_height_unchanged=True,waist_q_unchanged_at_equal_progress=True,phases_original_command_s=phases,time_scale=factor,duration_s=duration,samples=len(t),continuous_derivative_bounds=(bounds/[factor,factor**2,factor**3]).tolist(),fk_position_max_m=float(errors[:,:,0].max()),fk_orientation_max_deg=float(np.rad2deg(errors[:,:,1].max())),compliant_hand_collision_corridors=len(t),hardware_executed=False)
    report['pass']=report.pop('pass_')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out.resolve())
