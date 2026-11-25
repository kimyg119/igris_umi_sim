"""Plan a box transfer with the learned HL-IK prior.

Only the approach seed and caller-owned waist/lower-body reference are reused.
Every arm command is solved with PipelineIK and the real FiSTA checkpoint.
This file produces offline trajectories; it does not move a simulated robot.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from igris.hl_ik.common import matrix7,pose7,sha256
from igris.hl_ik.solver import ElbowPrior
from igris.hl_ik.pipeline import PipelineIK
from box_replay import load_motion

SOURCE=ROOT/'outputs/isaac_compliant_carry_20260918/lift10_forward8/motion_100hz.npz'
ORIGINAL=ROOT/'outputs/synthetic_box_motion_20260914'
ENVELOPE=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/compliance.json'
CHECKPOINT=ROOT/'outputs/hl_ik_20260915/training/full/seed_42/best.pt'


def smooth(value):
    u=np.clip(value,0,1)
    return u**3*(10-15*u+6*u*u)


def torso_world(solver,q,base):
    solver.set_q(q)
    body=solver.model.body('pelvis').id
    data=solver.configuration.data
    pelvis=matrix7(np.r_[data.xpos[body],data.xquat[body]])
    return base@np.linalg.inv(pelvis)


def build(out,model,*,destination=(.33,.20),lift=.17,reach=.08,squeeze=0.,waist_fraction=1.,
          duration_scale=1.,grasp_time=47.25,stop_time=None,limit_cost=10.,soft_margin=.15,
          grip_pitch_deg=0.,waist_pitch_offset=0.,place_drop=.06,follow_waist_yaw=False,
          grasp_center_normal=0.,grasp_open=0.,grasp_height_offset=0.,parallel_approach=False,
          grasp_center_radial=0.,pregrasp_waist_yaw_offset=0.,palm_first_roll_deg=(0.,0.),place_waist_pitch_offset=None,
          pregrasp_waist_pitch_offset=0.,joint_margin=.0001,lift_waist_pitch_offset=None):
    if lift_waist_pitch_offset is not None and not np.isfinite(lift_waist_pitch_offset):raise ValueError('Invalid lifting waist pitch')
    out.mkdir(parents=True,exist_ok=False)
    source=np.load(SOURCE);old=np.load(ORIGINAL/'motion_100hz.npz')
    prepared=np.load(ORIGINAL/'prepared.npz')
    raw=np.load(ORIGINAL/'raw_targets.npz')
    phases={str(raw['stage'][i]):float(np.interp(raw['t'][i]-old['source_offset_s'],old['source_time_s'],old['command_t_s']))
            for i in np.r_[0,np.flatnonzero(raw['stage'][1:]!=raw['stage'][:-1])+1]}
    metadata=json.loads((ORIGINAL/'source_metadata.json').read_text())
    base=matrix7(np.r_[metadata['base_position'],np.array(metadata['base_quat_xyzw'])[[3,0,1,2]]])
    solver=PipelineIK(ElbowPrior(CHECKPOINT),model_path=model,hand_envelope=ENVELOPE,
                      joint_limit_cost=limit_cost,soft_joint_margin=soft_margin,joint_margin=joint_margin)
    duration=float(source['command_t_s'][-1])
    ts=np.arange(int(np.floor(duration*30*duration_scale))+1)/30
    progress=ts/duration_scale
    source_time=np.interp(progress,source['command_t_s'],source['source_time_s'])
    reference=np.column_stack([np.interp(source_time,prepared['times'],prepared['q'][:,j]) for j in range(31)])
    owned=source['command_sdk_indices']
    reference[:,owned]=np.column_stack([np.interp(progress,source['command_t_s'],source['q_command_rad'][:,j]) for j in range(len(owned))])
    reference[:,29:]=0
    source_poses=source['ee_pose_torso']
    targets=np.empty((len(ts),2,7))
    for s in range(2):
        targets[:,s,:3]=np.column_stack([np.interp(progress,source['command_t_s'],source_poses[:,s,j]) for j in range(3)])
        targets[:,s,3:]=Slerp(source['command_t_s'],Rotation.from_quat(source_poses[:,s,3:],scalar_first=True))(progress).as_quat(scalar_first=True)
        # Choose palm in-plane orientation before contact. The local Y axis is
        # the opposing palm normal; this does not turn a grasped box.
        tilt=Rotation.from_rotvec(np.c_[np.zeros(len(ts)),np.deg2rad(grip_pitch_deg)*smooth(progress/phases['squeeze']),np.zeros(len(ts))])
        targets[:,s,3:]=(Rotation.from_quat(targets[:,s,3:],scalar_first=True)*tilt).as_quat(scalar_first=True)
        # A small mirrored local-X tilt brings the palm pad into contact before
        # the protruding compliant fingertip; this rotates only EE targets.
        roll=(1 if s==0 else -1)*np.deg2rad(palm_first_roll_deg[s])*smooth(progress/phases['squeeze'])
        tilt=Rotation.from_rotvec(np.c_[roll,np.zeros(len(ts)),np.zeros(len(ts))])
        targets[:,s,3:]=(Rotation.from_quat(targets[:,s,3:],scalar_first=True)*tilt).as_quat(scalar_first=True)
    anchor=int(np.argmin(abs(progress-grasp_time)))
    anchor_world=torso_world(solver,reference[anchor],base)
    hands=np.array([anchor_world@matrix7(p) for p in targets[anchor]])
    center=hands[:,:3,3].mean(0)
    direction=hands[0,:3,3]-hands[1,:3,3];direction/=np.linalg.norm(direction)
    radial=np.cross(direction,[0.,0.,1.]);radial/=np.linalg.norm(radial)
    # Correct the contact centre and clearance before first contact. Both are
    # EE translations; box poses, contact physics and finger limits are intact.
    for i in range(anchor):
        amount=smooth((progress[i]-20.)/(phases['squeeze']-20.))
        frame=torso_world(solver,reference[i],base)
        for side in range(2):
            goal=frame@matrix7(targets[i,side])
            goal[:3,3]+=direction*(grasp_center_normal+(1 if side==0 else -1)*grasp_open)*amount
            goal[:3,3]+=radial*grasp_center_radial*amount
            goal[2,3]+=grasp_height_offset*amount
            targets[i,side]=pose7(np.linalg.inv(frame)@goal)
    for side in range(2):
        hands[side,:3,3]+=direction*(grasp_center_normal+(1 if side==0 else -1)*grasp_open)
        hands[side,:3,3]+=radial*grasp_center_radial
        hands[side,2,3]+=grasp_height_offset
    if parallel_approach:
        first=int(np.argmin(abs(progress-20.)))
        initial_frame=torso_world(solver,reference[first],base)
        initial=np.array([initial_frame@matrix7(p) for p in targets[first]])
        seat=phases['seat_palm'];clearance=.055
        rotations=[Slerp([0.,1.],Rotation.from_matrix([initial[s,:3,:3],hands[s,:3,:3]])) for s in range(2)]
        for i in range(first,anchor):
            move=smooth((progress[i]-progress[first])/(seat-progress[first]))
            close=smooth((progress[i]-seat)/(grasp_time-seat))
            frame=torso_world(solver,reference[i],base)
            for side in range(2):
                goal=hands[side].copy()
                pregrasp=hands[side,:3,3]+direction*(1 if side==0 else -1)*clearance
                goal[:3,3]=(1-move)*initial[side,:3,3]+move*pregrasp
                goal[:3,3]-=direction*(1 if side==0 else -1)*clearance*close
                goal[:3,:3]=rotations[side](move).as_matrix()
                targets[i,side]=pose7(np.linalg.inv(frame)@goal)
    payload_xy=np.array([.37/np.sqrt(2),-.37/np.sqrt(2)])
    delta=np.asarray(destination)-payload_xy
    pivot=anchor_world[:2,3].copy()
    source_radius=payload_xy-pivot;destination_radius=np.asarray(destination)-pivot
    yaw_delta=float(np.arctan2(destination_radius[1],destination_radius[0])-np.arctan2(source_radius[1],source_radius[0]))
    yaw_delta=float(np.arctan2(np.sin(yaw_delta),np.cos(yaw_delta)))
    expected_yaw=np.zeros(len(ts))
    waist_anchor=reference[anchor,0]
    for i,t in enumerate(progress):
        if i<anchor:continue
        reference[i,0]=waist_anchor+waist_fraction*(reference[i,0]-waist_anchor)
        up=smooth((t-grasp_time)/(phases['carry_left']-grasp_time))
        carry=smooth((t-phases['carry_left'])/(phases['lower']-phases['carry_left']))
        lift_pitch=waist_pitch_offset if lift_waist_pitch_offset is None else lift_waist_pitch_offset
        reference[i,2]+=lift_pitch*up+(waist_pitch_offset-lift_pitch)*carry
        frame=torso_world(solver,reference[i],base)
        down=smooth((t-phases['lower'])/(phases['open']-phases['lower']))
        if place_waist_pitch_offset is not None:
            reference[i,2]+=(place_waist_pitch_offset-waist_pitch_offset)*down
            frame=torso_world(solver,reference[i],base)
        opening=smooth((t-phases['open'])/(duration-phases['open']))
        translation=np.r_[delta*carry,lift*up-(lift+place_drop)*down]
        translation[:2]+=reach*up*(1-down)*anchor_world[:2,0]
        rotation=np.eye(3)
        if follow_waist_yaw:
            # Rotate about the payload centre while the waist follows the same
            # arc. The pelvis remains floating and receives no position command.
            expected_yaw[i]=yaw_delta*carry
            rotation=Rotation.from_euler('z',expected_yaw[i]).as_matrix()
            reference[i,0]=waist_anchor-expected_yaw[i]
            frame=torso_world(solver,reference[i],base)
            angle=np.arctan2(source_radius[1],source_radius[0])+expected_yaw[i]
            radius=np.linalg.norm(source_radius)+(np.linalg.norm(destination_radius)-np.linalg.norm(source_radius))*carry
            radius+=reach*up*(1-down)
            translation[:2]=pivot+radius*np.array([np.cos(angle),np.sin(angle)])-payload_xy
        for s in range(2):
            goal=hands[s].copy();goal[:3,3]+=translation
            if follow_waist_yaw:
                goal[:3,:3]=rotation@hands[s,:3,:3]
                goal[:2,3]=payload_xy+translation[:2]+(rotation@(hands[s,:3,3]-np.r_[payload_xy,center[2]]))[:2]
            goal[:3,3]+=(1 if s==0 else -1)*(rotation@direction)*(-squeeze*up*(1-opening)+.055*opening)
            targets[i,s]=pose7(np.linalg.inv(frame)@goal)

    if pregrasp_waist_yaw_offset or pregrasp_waist_pitch_offset:
        # Caller-owned waist adjustment before contact, preserving both world
        # hand targets. This changes the arm posture, not the payload trajectory.
        for i,t in enumerate(progress):
            before=torso_world(solver,reference[i],base)
            reference[i,0]+=pregrasp_waist_yaw_offset*smooth((t-20.)/(phases['seat_palm']-20.))
            reference[i,2]+=pregrasp_waist_pitch_offset*smooth((t-20.)/(phases['seat_palm']-20.))*(1-smooth((t-grasp_time)/(phases['carry_left']-grasp_time)))
            after=torso_world(solver,reference[i],base)
            targets[i]=[pose7(np.linalg.inv(after)@before@matrix7(p)) for p in targets[i]]
    np.savez_compressed(out/'targets.npz',time=ts,reference=reference,targets=targets,expected_box_yaw_rad=expected_yaw)
    q=reference[0].copy();solver.reset(q)
    commands=[];statistics=[];failure=None;started=time.time()
    for i,(t,goal) in enumerate(zip(ts,targets)):
        if stop_time is not None and t>stop_time:break
        previous=q.copy();q[:15]=reference[i,:15]
        r=solver.solve(goal,q,record_history=False,position_tolerance=.003,orientation_tolerance=np.deg2rad(1))
        if r.accepted:
            solver.set_q(previous);a=solver.configuration.q.copy()
            solver.set_q(r.q);b=solver.configuration.q.copy()
            hit=solver.geometry.corridor(solver.configuration.data,a,b,max_step=.01)
        else:hit=None
        statistics.append([r.position_error_m,r.orientation_error_rad,r.elapsed_ms,r.accepted,r.success])
        commands.append(r.q.copy())
        if not r.accepted or hit or not r.success or r.prior_status!='used':
            failure=dict(frame=i,time_s=float(t),reason=r.reason,collision=hit,
                         prior_status=r.prior_status,position_m=r.position_error_m,orientation_deg=float(np.rad2deg(r.orientation_error_rad)))
            break
        q=r.q;solver.observe(q)
        if i%150==0:print('IK',i,len(ts),'t',round(t,2),'mm',round(r.position_error_m*1000,3),flush=True)
    qs=np.asarray(commands);stats=np.asarray(statistics)
    np.savez_compressed(out/'ik.npz',time=ts[:len(qs)],q=qs,goals=targets[:len(qs)],statistics=stats)
    report=dict(completed=failure is None and len(qs)==len(ts),failure=failure,frames=len(qs),
        requested_frames=len(ts),solver='igris.hl_ik.pipeline.PipelineIK',prior_sha256=sha256(CHECKPOINT),
        model_sha256=sha256(model),source_sha256=sha256(SOURCE),hardware_executed=False,
        phases=phases,grasp_time=grasp_time,lift_m=lift,reach_m=reach,extra_squeeze_m=squeeze,place_drop_m=place_drop,
        grasp_center_normal_m=grasp_center_normal,grasp_open_per_hand_m=grasp_open,
        grasp_height_offset_m=grasp_height_offset,parallel_pregrasp_approach=parallel_approach,
        grasp_center_radial_m=grasp_center_radial,
        pregrasp_waist_yaw_offset_rad=pregrasp_waist_yaw_offset,
        pregrasp_waist_pitch_offset_rad=pregrasp_waist_pitch_offset,
        lift_waist_pitch_offset_rad=lift_waist_pitch_offset,
        palm_first_roll_deg=list(palm_first_roll_deg),
        place_waist_pitch_offset=place_waist_pitch_offset,
        destination_xy=list(destination),waist_fraction=waist_fraction,waist_pitch_offset=waist_pitch_offset,anchor_world_hand_poses=hands.tolist(),
        joint_limit_preference_cost=limit_cost,soft_joint_margin=soft_margin,grip_pitch_deg=grip_pitch_deg,
        joint_margin_rad=joint_margin,
        target_rotation='follow_waist_yaw' if follow_waist_yaw else 'world_fixed',
        expected_box_yaw_change_rad=float(expected_yaw[-1]),
        rotation_pivot_xy=pivot.tolist(),
        wall_seconds=time.time()-started)
    (out/'plan_result.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['completed']:
        print(json.dumps(report,indent=2));return report
    spline=CubicSpline(ts,qs,axis=0,bc_type=((1,np.zeros(31)),(1,np.zeros(31))))
    dense=np.linspace(0,ts[-1],int(ts[-1]*1000)+1)
    bounds=[float(abs(spline(dense,n)[:,owned]).max()) for n in (1,2,3)]
    factor=max(1,bounds[0]/.49,np.sqrt(bounds[1]/3),np.cbrt(bounds[2]/100))*1.002
    end=np.ceil(ts[-1]*factor*100)/100;factor=end/ts[-1]
    t100=np.arange(round(end*100)+1)/100;source_t=t100/factor
    command=spline(source_t);last=None
    for i,cur in enumerate(command):
        solver.set_q(cur);b=solver.configuration.q.copy();arm=cur[15:29]
        limited=solver.model.jnt_limited.astype(bool)
        positions=b[solver.model.jnt_qposadr[limited]]
        if np.any(positions<solver.model.jnt_range[limited,0]-1e-8) or np.any(positions>solver.model.jnt_range[limited,1]+1e-8):
            raise ValueError(f'Interpolated full-body joint limits {i}')
        if np.any(arm<solver.limits[:,0]) or np.any(arm>solver.limits[:,1]) or np.any(solver.A@arm>solver.b+1e-8):
            raise ValueError(f'Interpolated joint limits {i}')
        hit=solver.geometry.corridor(solver.configuration.data,b if last is None else last,b,max_step=.005)
        if hit:raise ValueError(f'Interpolated collision {i}: {hit}')
        last=b
    original_progress=np.interp(source_t,ts,progress)
    original_command=np.interp(original_progress,source['command_t_s'],source['original_command_t_s'])
    np.savez_compressed(out/'motion_100hz.npz',command_t_s=t100,q_command_rad=command[:,owned],
        command_sdk_indices=owned,mode=np.full(len(t100),'FOLLOWING'),trip=np.full(len(t100),''),
        original_command_t_s=original_command,source_time_s=np.interp(original_progress,source['command_t_s'],source['source_time_s']),
        source_offset_s=source['source_offset_s'],frame='torso',rate_hz=100.,
        expected_box_yaw_rad=np.interp(source_t,ts,expected_yaw),
        target_rotation='follow_waist_yaw' if follow_waist_yaw else 'world_fixed',
        motion_knot_t_s=ts*factor,motion_knot_q_rad=qs[:,owned])
    load_motion(out/'motion_100hz.npz')
    report.update(exported=True,duration_s=end,time_scale=factor,
                  derivative_bounds=[bounds[i]/factor**(i+1) for i in range(3)],
                  motion_sha256=sha256(out/'motion_100hz.npz'))
    (out/'plan_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--destination',type=float,nargs=2,default=(.33,.20))
    p.add_argument('--lift',type=float,default=.17);p.add_argument('--squeeze',type=float,default=0.)
    p.add_argument('--reach',type=float,default=.08)
    p.add_argument('--waist-fraction',type=float,default=1.);p.add_argument('--duration-scale',type=float,default=1.)
    p.add_argument('--grasp-time',type=float,default=47.25);p.add_argument('--stop-time',type=float)
    p.add_argument('--limit-cost',type=float,default=10.)
    p.add_argument('--soft-margin',type=float,default=.15)
    p.add_argument('--joint-margin',type=float,default=.0001)
    p.add_argument('--grip-pitch',type=float,default=0.)
    p.add_argument('--waist-pitch-offset',type=float,default=0.)
    p.add_argument('--lift-waist-pitch',type=float,help='Pitch at lift completion; blend to carry pitch while turning')
    p.add_argument('--place-drop',type=float,default=.06)
    p.add_argument('--follow-waist-yaw',action='store_true')
    p.add_argument('--grasp-center-normal',type=float,default=0.)
    p.add_argument('--grasp-open',type=float,default=0.)
    p.add_argument('--grasp-height-offset',type=float,default=0.)
    p.add_argument('--parallel-approach',action='store_true')
    p.add_argument('--grasp-center-radial',type=float,default=0.)
    p.add_argument('--pregrasp-waist-yaw-offset',type=float,default=0.)
    p.add_argument('--pregrasp-waist-pitch-offset',type=float,default=0.)
    p.add_argument('--palm-first-roll',type=float,nargs=2,default=(0.,0.))
    p.add_argument('--place-waist-pitch',type=float)
    a=p.parse_args();build(a.out.resolve(),a.model.resolve(),destination=a.destination,lift=a.lift,reach=a.reach,
        squeeze=a.squeeze,waist_fraction=a.waist_fraction,duration_scale=a.duration_scale,
        grasp_time=a.grasp_time,stop_time=a.stop_time,limit_cost=a.limit_cost,soft_margin=a.soft_margin,
        grip_pitch_deg=a.grip_pitch,waist_pitch_offset=a.waist_pitch_offset,place_drop=a.place_drop,
        follow_waist_yaw=a.follow_waist_yaw,grasp_center_normal=a.grasp_center_normal,grasp_open=a.grasp_open,
        grasp_height_offset=a.grasp_height_offset,parallel_approach=a.parallel_approach,
        grasp_center_radial=a.grasp_center_radial,pregrasp_waist_yaw_offset=a.pregrasp_waist_yaw_offset,
        palm_first_roll_deg=a.palm_first_roll,place_waist_pitch_offset=a.place_waist_pitch,
        pregrasp_waist_pitch_offset=a.pregrasp_waist_pitch_offset,joint_margin=a.joint_margin,lift_waist_pitch_offset=a.lift_waist_pitch)
