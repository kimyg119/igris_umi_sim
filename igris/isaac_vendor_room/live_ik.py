"""Host-side HL-IK worker for a private native-simulation command publisher.

World targets come from a validated plan. Native torso poses compensate floating
base motion. This process only writes arm references; the existing isolated SDK
publisher owns commands and all motor/backlash checks.
"""
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
from collections import deque
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from igris.hl_ik.common import matrix7,pose7,sha256
from igris.hl_ik.solver import ElbowPrior
from igris.hl_ik.pipeline import PipelineIK
from transport_plan import CHECKPOINT,ENVELOPE,ORIGINAL,torso_world
from reference_spline import evaluate as spline_reference
from grasp_feedback import GraspFeedback
from box_orientation_feedback import BoxOrientationFeedback,rotate_target
from transport_trial import htd_pair_buffer_overrides


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,allow_nan=False))
    temporary.replace(path)


def primary_qp_start(manifest):
    """Cold primary is an explicit pinned transport scope; other callers stay warm."""
    scope=manifest.get('htd_loaded_zero') or {}
    if scope.get('scope')!='loaded_transport':return 'warm'
    if (scope.get('enabled') is not True
            or (scope.get('transport') or {}).get('primary_qp_start')!='cold'):
        raise ValueError('HTD loaded transport requires explicit cold primary QP selection')
    return 'cold'


def primary_qp_iteration_limits(manifest):
    """Only the reviewed transport scope shortens primary work before fallback."""
    scope=manifest.get('htd_loaded_zero') or {}
    if scope.get('scope')!='loaded_transport':return {'outer':50,'inner':50}
    primary_qp_start(manifest)
    limits=(scope.get('transport') or {}).get('primary_qp_iteration_limits')
    if (not isinstance(limits,dict) or set(limits)!={'outer','inner'}
            or any(type(value) is not int or value!=10 for value in limits.values())):
        raise ValueError('HTD loaded transport requires primary QP iteration limits 10/10')
    return dict(limits)


def run(out,plan,model,acceleration_limit=3.,velocity_limit=.49,torso_filter_seconds=.4,force_grasp=False,
        joint_limit_cost=None,soft_joint_margin=None,collision_braking=False,track_source=False,lead_steps=3,
        orientation_feedback=False,grasp_travel=.015,track_source_rotation=False,synchronized_grasp=False,box_orientation_feedback=False,native_collision_contacts=False,grasp_height_adjustment=0.):
    if not np.isfinite(torso_filter_seconds) or torso_filter_seconds<0:raise ValueError('Invalid torso filter time constant')
    if not isinstance(lead_steps,int) or not 3<=lead_steps<=9:raise ValueError('Scheduled lead must be 3 to 9 native IK ticks')
    if not np.isfinite(grasp_travel) or not .005<=grasp_travel<=.03:raise ValueError('EE preload travel must be 5 to 30 mm')
    if not np.isfinite(grasp_height_adjustment) or not -.06<=grasp_height_adjustment<=.04:
        raise ValueError('Grasp height adjustment outside the bounded target range')
    if orientation_feedback and not force_grasp:raise ValueError('Orientation feedback requires contact force feedback')
    if synchronized_grasp and not force_grasp:raise ValueError('Synchronized grasp requires contact force feedback')
    if box_orientation_feedback and not force_grasp:raise ValueError('Payload orientation feedback requires force-qualified grasp')
    if track_source_rotation and not track_source:raise ValueError('Source rotation tracking requires source translation tracking')
    manifest=json.loads((out/'manifest.json').read_text())
    pair_overrides=htd_pair_buffer_overrides(manifest)
    qp_start=primary_qp_start(manifest)
    qp_limits=primary_qp_iteration_limits(manifest)
    pipeline_source_sha256=sha256(ROOT/'igris/hl_ik/pipeline.py')
    from lowering_targets import phase_acceleration_limit
    transfer_spec=manifest.get('inplace_transfer',manifest.get('walking_transfer',{}))
    lowering_acceleration=transfer_spec.get('lowering_acceleration_limit')
    phase_acceleration_limit(acceleration_limit,lowering_acceleration,None)
    if lowering_acceleration is not None and (manifest.get('evaluation_profile')!='relaxed_transport'
            or not transfer_spec.get('world_position_lowering',False)):
        raise ValueError('Lowering acceleration override requires explicit relaxed world-position lowering')
    if not manifest.get('stage_room_after_preparation'):
        raise ValueError('The world-target transform requires staged room initialization')
    report=json.loads((plan/'plan_result.json').read_text())
    if not report.get('exported'):raise ValueError('A fully validated offline plan is required')
    targets=np.load(plan/'targets.npz');ik=np.load(plan/'ik.npz')
    limit_cost=report.get('joint_limit_preference_cost',0.) if joint_limit_cost is None else joint_limit_cost
    soft_margin=report.get('soft_joint_margin',.15) if soft_joint_margin is None else soft_joint_margin
    solver=PipelineIK(ElbowPrior(CHECKPOINT),model_path=model,hand_envelope=ENVELOPE,
                      acceleration_limit=acceleration_limit,
                      joint_limit_cost=limit_cost,soft_joint_margin=soft_margin,
                      collision_braking=collision_braking,warm_start_primary=qp_start=='warm',
                      primary_max_iter=qp_limits['outer'],primary_max_iter_in=qp_limits['inner'],
                      pair_buffer_overrides=pair_overrides)
    if native_collision_contacts:solver.geometry.enable_native_contacts()
    collision_geometry=solver.geometry.provenance()
    if pair_overrides is not None and collision_geometry.get('pair_buffer_overrides')!=[
            dict(pair=list(pair),buffer_per_sphere_m=value,gap_offset_m=.005)
            for pair,value in pair_overrides.items()]:
        raise ValueError('HL-IK pair padding was not applied to the solver geometry')
    metadata=json.loads((ORIGINAL/'source_metadata.json').read_text())
    base=matrix7(np.r_[metadata['base_position'],np.array(metadata['base_quat_xyzw'])[[3,0,1,2]]])
    world=np.empty((len(targets['time']),2,7))
    for i,(reference,pair) in enumerate(zip(targets['reference'],targets['targets'])):
        torso=torso_world(solver,reference,base)
        world[i]=[pose7(torso@matrix7(p)) for p in pair]
    times=targets['time']*report['time_scale']
    grasp_requirements=manifest.get('grasp_requirements',{})
    feedback=GraspFeedback(manifest['payload']['mass_kg'],manifest['payload']['friction'],travel=grasp_travel,synchronized=synchronized_grasp,
                          target=grasp_requirements.get('target_force_N',14.),tip_min_force=grasp_requirements.get('tip_min_force_N',.2)) if force_grasp else None
    box_feedback=BoxOrientationFeedback() if box_orientation_feedback else None
    grasp_time=report['grasp_time']*report['time_scale'];open_time=report['phases']['open']*report['time_scale']
    grasp_verified=False;grasp_wait_started=None;grasp_verified_at=None
    layout=json.loads((out.parent/'layout.json').read_text())
    source_xyz=np.asarray(layout['boxes'][manifest['payload']['usd_path'].rsplit('/',1)[-1]]['xyz_m'])
    source_offset=np.zeros(3)
    source_rotation=np.eye(3)
    source_box=layout['boxes'][manifest['payload']['usd_path'].rsplit('/',1)[-1]]
    nominal_box_rotation=(Rotation.from_euler('z',source_box['yaw_deg'],degrees=True)*Rotation.from_euler('x',source_box['roll_deg'],degrees=True)).as_matrix()
    anchor_hand_center=np.asarray(report['anchor_world_hand_poses'])[:,:3,3].mean(0)
    contact_width=np.full(2,np.nan);closure_opening=np.zeros(2)
    contact_start=report['phases']['seat_palm']*report['time_scale']
    rotations=[Slerp(times,Rotation.from_quat(world[:,side,3:],scalar_first=True)) for side in range(2)]
    nominal=CubicSpline(ik['time']*report['time_scale'],ik['q'],axis=0,
                        bc_type=((1,np.zeros(31)),(1,np.zeros(31))))
    # Defer cyclic GC during this bounded trial to avoid pauses during command
    # production; restore GC on exit and keep every solver gate active.
    gc_was_enabled=gc.isenabled();gc.collect();gc.disable()
    atomic_json(out/'live_ik_ready.json',dict(prior_sha256=sha256(CHECKPOINT),
        model_sha256=sha256(model),motion_sha256=report['motion_sha256'],native_collision_contacts=native_collision_contacts,
        primary_qp_start=qp_start,primary_qp_iteration_limits=qp_limits,pipeline_source_sha256=pipeline_source_sha256,
        collision_geometry=collision_geometry,
        native_contacts_sha256=solver.geometry._native_contacts.source_sha256 if native_collision_contacts else None,hardware_executed=False))
    stopped=False;last=None;q=None;stage=None;rows=[];error=None;heartbeat_at=0.;knots=[];filtered_torso=None;cycle_ms=[]
    segment_history=deque(maxlen=8)
    walking_preload=None;world_position_lowering=None
    def stop(*args):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        with (out/'live_ik_ticks.jsonl').open('w') as log:
            while not stopped:
                if time.time()-heartbeat_at>.05:
                    heartbeat_at=time.time()
                    atomic_json(out/'live_ik_heartbeat.json',dict(wall=heartbeat_at))
                request_path=out/'live_ik_request.json'
                if not request_path.exists():time.sleep(.005);continue
                req=json.loads(request_path.read_text())
                walking=req.get('walking')
                if not req['active']:time.sleep(.005);continue
                effective_acceleration=phase_acceleration_limit(acceleration_limit,lowering_acceleration,walking)
                solver.acceleration_limit=effective_acceleration
                if (out/'stop.request').exists():break
                if time.time()-req['wall']>.2:raise RuntimeError('Stale SDK/torso request')
                t=float(req['motion_time'])
                control=float(req.get('control_time',t))
                tick=control if last is None else last+1/30
                if control+1e-9<tick:time.sleep(.002);continue
                if control-tick>.10:raise RuntimeError('HL-IK simulation cadence exceeded 100 ms')
                cycle_started=time.perf_counter()
                if stage is None:
                    placement=json.loads((out/'room_layout_applied.json').read_text())
                    stage=np.eye(4);stage[:3,:3]=Rotation.from_euler('z',placement['yaw_rad']).as_matrix()
                    stage[:2,3]=placement['translation_xy']
                next_t=min(t+1/30,times[-1]);reference=nominal(next_t)
                from htd_loaded_reference import apply_loaded_waist
                reference=apply_loaded_waist(reference,req,
                    enabled=manifest.get('htd_loaded_zero',{}).get('enabled',False))
                if walking is not None and manifest.get('inplace_transfer'):
                    from inplace_reference import apply_waist_reference
                    reference=apply_waist_reference(reference,walking)
                    if not np.array_equal(np.asarray(req['nominal'])[:3],reference[:3]):
                        raise RuntimeError('SDK and HL-IK caller waist references differ')
                if q is None:
                    q=np.asarray(req['nominal'],float);q[3:15]=req['measured'][3:15];q[29:]=0
                    solver.reset(q)
                    knots=[q[15:29].copy() for _ in range(3)]
                previous=q.copy();q[:3]=reference[:3];q[3:15]=req['measured'][3:15];q[29:]=0
                measured_torso=matrix7(req['torso_pose'])
                if filtered_torso is None:filtered_torso=measured_torso.copy()
                alpha=1. if torso_filter_seconds==0 else 1.-np.exp(-(1/30)/torso_filter_seconds)
                filtered_torso[:3,3]+=alpha*(measured_torso[:3,3]-filtered_torso[:3,3])
                filtered_torso[:3,:3]=Slerp([0,1],Rotation.from_matrix([filtered_torso[:3,:3],measured_torso[:3,:3]]))(alpha).as_matrix()
                inverse=np.linalg.inv(filtered_torso)
                # Blend frame correction before first contact; no discontinuous
                # pose jump when the live controller takes ownership.
                u=np.clip(control/8.,0,1);blend=u**3*(10-15*u+6*u*u)
                nominal_torso=torso_world(solver,reference,base)
                force_sample=None
                if feedback is not None and t>=contact_start:
                    force_sample=json.loads((out/'grasp_force.json').read_text())
                    if not 0<=time.time()-force_sample['wall']<.25:raise RuntimeError('Stale native payload contact forces')
                    if t<open_time and (walking is None or walking['phase'] not in ('open','done')):
                        separation=np.array([np.interp(next_t,times,world[:,0,j]-world[:,1,j]) for j in range(3)])
                        half_width=np.linalg.norm(separation)/2
                        forces=np.asarray(force_sample['palm_tip_force_N']).reshape(2,2)
                        newly_contacted=np.isnan(contact_width)&(forces.sum(1)>=.5)
                        contact_width[newly_contacted]=half_width
                        active=np.isfinite(contact_width)
                        if t<grasp_time:
                            # Once a hand touches, cancel the remaining nominal
                            # closing motion. Only bounded force feedback adds
                            # preload; the full planned approach cannot crush
                            # the compliant fingers before feedback starts.
                            closure_opening[active]=np.maximum(0.,contact_width[active]-half_width)
                        _,ready=feedback.update(force_sample['palm_tip_force_N'],1/30,active=active)
                        if t>=grasp_time:
                            if grasp_wait_started is None:grasp_wait_started=control
                            if ready and not grasp_verified:grasp_verified=True;grasp_verified_at=control
                            if not grasp_verified and control-grasp_wait_started>20:
                                raise RuntimeError('No friction-qualified grasp within 20 simulation seconds')
                normal=np.array([np.interp(next_t,times,world[:,0,j]-world[:,1,j]) for j in range(3)])
                normal/=np.linalg.norm(normal)
                if track_source and t<grasp_time and not np.isfinite(contact_width).any():
                    # Track the settling source box only before lift. Move EE
                    # targets, never object state. Freeze at grasp; fade this
                    # correction during carry so receiving targets stay fixed.
                    observed=np.linalg.inv(stage)@np.r_[req['payload_pose'][:3],1.]
                    desired_offset=observed[:3]-source_xyz
                    if not np.isfinite(desired_offset).all() or np.linalg.norm(desired_offset)>.08:
                        raise RuntimeError('Source box moved outside 80 mm acquisition range')
                    update=(1.-np.exp(-(1/30)/.5))*(desired_offset-source_offset)
                    source_offset+=update*min(1.,(.003/30)/max(np.linalg.norm(update),1e-12))
                    if track_source_rotation:
                        observed_rotation=stage[:3,:3].T@matrix7(req['payload_pose'])[:3,:3]
                        desired_rotation=observed_rotation@nominal_box_rotation.T
                        if Rotation.from_matrix(desired_rotation).magnitude()>np.deg2rad(5):
                            raise RuntimeError('Source box rotated outside 5 degree acquisition range')
                        rotation_step=(1.-np.exp(-(1/30)/.5))*Rotation.from_matrix(source_rotation.T@desired_rotation).as_rotvec()
                        rotation_step*=min(1.,(np.deg2rad(.5)/30)/max(np.linalg.norm(rotation_step),1e-12))
                        source_rotation=source_rotation@Rotation.from_rotvec(rotation_step).as_matrix()
                carry_start=report['phases']['carry_left']*report['time_scale']
                carry_end=report['phases']['lower']*report['time_scale']
                carry_fraction=np.clip((t-carry_start)/(carry_end-carry_start),0,1)
                source_weight=1.-carry_fraction**3*(10-15*carry_fraction+6*carry_fraction**2)
                yaw=float(np.interp(next_t,times,targets['expected_box_yaw_rad']))
                yaw_rotation=Rotation.from_euler('z',yaw).as_matrix()
                applied_source_rotation=yaw_rotation@Rotation.from_rotvec(Rotation.from_matrix(source_rotation).as_rotvec()*source_weight).as_matrix()@yaw_rotation.T
                center=source_xyz
                if t>=grasp_time:
                    hand_center=np.array([[np.interp(next_t,times,world[:,side,j]) for j in range(3)] for side in range(2)]).mean(0)
                    center=hand_center-yaw_rotation@(anchor_hand_center-source_xyz)
                normal=applied_source_rotation@normal
                release=np.clip((t-open_time)/max(times[-1]-open_time,1e-6),0,1)
                release=release**3*(10-15*release+6*release*release)
                box_correction=np.eye(3);box_feedback_active=False
                if box_feedback is not None:
                    expected_box=yaw_rotation@source_rotation@nominal_box_rotation
                    observed_box=stage[:3,:3].T@matrix7(req['payload_pose'])[:3,:3]
                    box_feedback_active=bool(walking is None and grasp_verified and t<open_time and np.all(feedback.filtered[[0,2]]>=2.))
                    box_correction=box_feedback.update(expected_box,observed_box,yaw,1/30,box_feedback_active)
                    box_correction=Rotation.from_rotvec(Rotation.from_matrix(box_correction).as_rotvec()*(1-release)).as_matrix()
                    normal=box_correction@normal
                measured_roll_error=None
                if orientation_feedback and walking is None and force_sample is not None and t<open_time:
                    measured_q=np.asarray(req['measured'],float).copy();measured_q[29:]=0
                    solver.set_q(measured_q)
                    measured_roll_error=[]
                    for side,task in enumerate(solver.tasks[:2]):
                        actual=measured_torso@solver.configuration.get_transform_frame_to_world(task.frame_name,'body').as_matrix()
                        desired_rotation=stage[:3,:3]@box_correction@applied_source_rotation@rotations[side](next_t).as_matrix()
                        measured_roll_error.append(float(Rotation.from_matrix(desired_rotation.T@actual[:3,:3]).as_rotvec()[0]))
                    feedback.update_roll(measured_roll_error,1/30,feedback.filtered[[0,2]]>=2.)
                    solver.set_q(q)
                goals=[]
                for side in range(2):
                    p=np.r_[[np.interp(next_t,times,world[:,side,j]) for j in range(3)],
                            rotations[side](next_t).as_quat(scalar_first=True)]
                    planned=matrix7(p)
                    planned[:3,3]=center+applied_source_rotation@(planned[:3,3]-center)
                    planned[:3,:3]=applied_source_rotation@planned[:3,:3]
                    if orientation_feedback:
                        planned[:3,:3]=planned[:3,:3]@Rotation.from_euler('x',feedback.roll_offset[side]*(1-release)).as_matrix()
                    planned[:3,3]+=source_weight*source_offset
                    if box_feedback is not None:
                        planned=rotate_target(planned,center+source_weight*source_offset,box_correction)
                    # Move the EE goals before contact to seat the palm pad
                    # farther down the box. Ramp before acquisition; preserve
                    # object state and the captured offset during later carry.
                    planned[2,3]+=grasp_height_adjustment*blend
                    if feedback is not None:planned[:3,3]-=(1 if side==0 else -1)*normal*(feedback.offset[side]-closure_opening[side])*(1-release)
                    corrected=inverse@stage@planned
                    original=np.linalg.inv(nominal_torso)@planned
                    original[:3,3]=(1-blend)*original[:3,3]+blend*corrected[:3,3]
                    original[:3,:3]=Slerp([0,1],Rotation.from_matrix([original[:3,:3],corrected[:3,:3]]))(blend).as_matrix()
                    goals.append(pose7(original))
                if walking is not None:
                    if feedback is None or not grasp_verified:raise RuntimeError('Walking goals require qualified live grasp')
                    if walking_preload is None:walking_preload=feedback.offset.copy()
                    pair=np.asarray(walking['world_hand_targets'],float)
                    if pair.shape!=(2,4,4) or not np.isfinite(pair).all():raise ValueError('Invalid walking EE goals')
                    normal=pair[0,:3,3]-pair[1,:3,3];normal/=np.linalg.norm(normal)
                    release=float(walking['release_fraction'])
                    goals=[]
                    for side in range(2):
                        relative=walking.get('torso_relative_hand_targets')
                        goal=np.asarray(relative[side],float).copy() if relative is not None else inverse@pair[side]
                        goal_normal=normal if relative is None else (
                            np.asarray(relative[0])[:3,3]-np.asarray(relative[1])[:3,3])
                        if relative is None:goal_normal=inverse[:3,:3]@goal_normal
                        goal_normal/=np.linalg.norm(goal_normal)
                        # Capture already includes preload. Only subsequent
                        # force corrections are added, then removed at release.
                        # During carry the reference is constant in the torso
                        # frame; do not introduce the torso filter's world lag.
                        goal[:3,3]-=(1 if side==0 else -1)*goal_normal*(feedback.offset[side]-walking_preload[side])*(1-release)
                        goals.append(pose7(goal))
                    if walking.get('world_position_lowering',False) and walking['phase'] in ('lower','open','done'):
                        from lowering_targets import WorldPositionLowering
                        relative=np.asarray(walking['torso_relative_hand_targets'],float)
                        current_pair=np.asarray([matrix7(goal) for goal in goals])
                        preload=current_pair[:,:3,3]-relative[:,:3,3]
                        if world_position_lowering is None:
                            if not rows or rows[-1].get('walking',{}).get('phase') not in ('carry','waist_arc'):
                                raise RuntimeError('World-position lowering needs the accepted carry boundary')
                            captured=np.asarray([matrix7(goal) for goal in rows[-1]['goals']])
                            prior_relative=np.asarray(rows[-1]['walking']['torso_relative_hand_targets'],float)
                            world_position_lowering=WorldPositionLowering(captured,matrix7(rows[-1]['filtered_torso_pose']),
                                walking['lowering_delta_world_m'],captured[:,:3,3]-prior_relative[:,:3,3])
                        pair=world_position_lowering.targets(filtered_torso,walking['lowering_fraction'],release,preload,
                            xy_fraction=walking.get('lowering_xy_fraction'))
                        goals=[pose7(goal) for goal in pair]
                result=solver.solve(goals,q,record_history=False,velocity_limit=velocity_limit,
                                    position_tolerance=.003,orientation_tolerance=np.deg2rad(1))
                row=dict(wall=time.time(),motion_time=t,control_time=control,accepted=result.accepted,
                         scheduled_tick_time=tick,control_lag_s=control-tick,
                         acceleration_limit_rad_s2=effective_acceleration,
                         prior_status=result.prior_status,position_error_m=result.position_error_m,
                         orientation_error_deg=float(np.rad2deg(result.orientation_error_rad)),
                         reason=result.reason,solve_ms=result.elapsed_ms,
                         qp_ms=result.qp_ms,collision_ms=result.collision_ms,
                         qp_fallbacks=result.qp_fallbacks,iterations=result.iterations,
                         velocity_max_rad_s=float(abs(solver.velocity).max()),
                         joint_margin_rad=float(np.min(np.r_[result.q[15:29]-solver.limits[:,0],solver.limits[:,1]-result.q[15:29],solver.b-solver.A@result.q[15:29]])),
                         q=result.q.tolist(),goals=np.asarray(goals).tolist(),torso_pose=req['torso_pose'],
                         filtered_torso_pose=pose7(filtered_torso).tolist())
                if track_source:row.update(source_offset_m=source_offset.tolist(),payload_pose=req['payload_pose'])
                if track_source_rotation:row['source_rotation_wxyz']=Rotation.from_matrix(source_rotation).as_quat(scalar_first=True).tolist()
                if feedback is not None:row.update(grasp_verified=grasp_verified,
                    grasp_offset_m=feedback.offset.tolist(),closure_opening_m=closure_opening.tolist(),
                    contact_latched=np.isfinite(contact_width).tolist(),payload_force_sample=force_sample)
                if orientation_feedback:row.update(measured_roll_error_rad=measured_roll_error,roll_offset_rad=feedback.roll_offset.tolist())
                if feedback is not None:row.update(grasp_target_force_N=feedback.commanded_target,paired_contact_seconds=feedback.paired_time)
                if box_feedback is not None:row.update(box_orientation_error_deg=box_feedback.error_deg,
                    box_correction_rotvec_rad=box_feedback.correction.as_rotvec().tolist(),box_feedback_active=box_feedback_active)
                if walking is not None:row['walking']=walking
                log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
                if not result.accepted or result.prior_status!='used':
                    raise RuntimeError('HL-IK gate: '+result.reason)
                before_contact=t<report['phases']['squeeze']*report['time_scale']
                maximum_error=.05 if before_contact else .015
                if result.position_error_m>maximum_error or result.orientation_error_rad>np.deg2rad(4):
                    raise RuntimeError(f'HL-IK live tracking error exceeded {maximum_error*1000:g} mm / 4 degrees')
                # The arm solver leaves caller-owned axes alone. Audit their
                # combined corridor too, including measured leg motion.
                solver.set_q(previous);a=solver.configuration.q.copy()
                solver.set_q(result.q);b=solver.configuration.q.copy()
                hit=solver.geometry.corridor(solver.configuration.data,a,b,max_step=.005)
                if hit:raise RuntimeError('HL-IK full corridor: '+str(hit))
                q=result.q;solver.observe(q)
                knots=(knots+[q[15:29].copy()])[-4:]
                # A cubic B-spline is C2. Its velocity and acceleration are
                # convex combinations of bounded finite differences, avoiding
                # the acceleration impulses of piecewise linear interpolation.
                previous_configuration=None
                # Derivatives of a uniform cubic B-spline are convex blends of
                # these finite differences, so this bounds the whole curve.
                differences=np.diff(np.asarray(knots),axis=0)*30
                if np.max(abs(differences))>velocity_limit+1e-6 or np.max(abs(np.diff(differences,axis=0)*30))>effective_acceleration+1e-4:
                    raise RuntimeError('Interpolated HL-IK rate bound')
                for u in (0.,.5,1.):
                    arm=spline_reference(knots,u)
                    sampled=q.copy();sampled[15:29]=arm;solver.set_q(sampled)
                    configuration=solver.configuration.q.copy()
                    hit=solver.geometry.corridor(solver.configuration.data,
                        configuration if previous_configuration is None else previous_configuration,configuration,max_step=.005)
                    if hit:raise RuntimeError('Interpolated HL-IK corridor: '+str(hit))
                    previous_configuration=configuration
                # Schedule complete segments ahead of their execution. Starting
                # at the measurement timestamp caused reference jumps when the
                # solve arrived halfway through the preceding interpolation.
                response=dict(wall=time.time(),start_time=tick+lead_steps/30,
                    end_time=tick+(lead_steps+1)/30,previous=previous[15:29].tolist(),target=q[15:29].tolist(),
                    control_points=np.asarray(knots).tolist(),interpolation='uniform_cubic_bspline',
                    position_error_m=result.position_error_m,orientation_error_rad=result.orientation_error_rad,
                    grasp_gate_time=grasp_time if feedback is not None else None,grasp_verified=grasp_verified,
                    world_hand_targets=[(filtered_torso@matrix7(p)).tolist() for p in goals],
                    torso_relative_hand_targets=[matrix7(p).tolist() for p in goals],
                    walking_phase=None if walking is None else walking['phase'],
                    reference_control_time_s=float(req['control_time']),
                    accepted=True,prior_status='used')
                segment_history.append(response)
                atomic_json(out/'live_ik_response.json',dict(response,segments=list(segment_history)))
                cycle_ms.append((time.perf_counter()-cycle_started)*1000)
                last=tick
    except Exception as exc:
        error=str(exc);atomic_json(out/'live_ik_error.json',dict(error=error,wall=time.time()))
        (out/'stop.request').write_text('live_hl_ik: '+error)
    finally:
        if gc_was_enabled:gc.enable()
        atomic_json(out/'live_ik_result.json',dict(error=error,ticks=len(rows),
            primary_qp_start=qp_start,primary_qp_iteration_limits=qp_limits,pipeline_source_sha256=pipeline_source_sha256,
            all_accepted=bool(rows) and all(r['accepted'] and r['prior_status']=='used' for r in rows),
            position_error_max_m=max((r['position_error_m'] for r in rows),default=None),
            orientation_error_max_deg=max((r['orientation_error_deg'] for r in rows),default=None),
            prior_sha256=sha256(CHECKPOINT),velocity_limit=velocity_limit,
            acceleration_limit=acceleration_limit,lowering_acceleration_limit=lowering_acceleration,scheduled_lead_s=lead_steps/30,
            torso_filter_seconds=torso_filter_seconds,synchronized_grasp=synchronized_grasp,
            force_control_enabled=force_grasp,grasp_verified_at_control_s=grasp_verified_at,
            orientation_feedback=orientation_feedback,grasp_travel_m=grasp_travel,
            grasp_height_adjustment_m=grasp_height_adjustment,
            box_orientation_feedback=box_orientation_feedback,
            source_rotation_tracking=track_source_rotation,source_rotation_wxyz=Rotation.from_matrix(source_rotation).as_quat(scalar_first=True).tolist(),
            joint_limit_preference_cost=limit_cost,soft_joint_margin=soft_margin,
            collision_braking=collision_braking,
            collision_geometry=collision_geometry,
            native_collision_contacts=native_collision_contacts,
            native_contact_tie_fallbacks=solver.geometry._native_contacts.tie_fallbacks if native_collision_contacts else None,
            source_tracking_enabled=track_source,source_offset_m=source_offset.tolist(),
            closure_opening_m=closure_opening.tolist(),contact_latched=np.isfinite(contact_width).tolist(),
            palm_force_required_N=feedback.required if feedback is not None else None,
            grasp_requirements=grasp_requirements,evaluation_profile=manifest.get('evaluation_profile','original_grasp_requirements'),
            cycle_ms_p95=float(np.percentile(cycle_ms,95)) if cycle_ms else None,
            cycle_ms_max=max(cycle_ms,default=None),hardware_executed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','plan','model'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--acceleration-limit',type=float,default=3.)
    p.add_argument('--velocity-limit',type=float,default=.49)
    p.add_argument('--torso-filter-seconds',type=float,default=.4)
    p.add_argument('--force-grasp',action='store_true')
    p.add_argument('--joint-limit-cost',type=float)
    p.add_argument('--soft-joint-margin',type=float)
    p.add_argument('--collision-braking',action='store_true')
    p.add_argument('--track-source',action='store_true')
    p.add_argument('--lead-steps',type=int,default=3)
    p.add_argument('--orientation-feedback',action='store_true')
    p.add_argument('--grasp-travel',type=float,default=.015)
    p.add_argument('--grasp-height-adjustment',type=float,default=0.)
    p.add_argument('--synchronized-grasp',action='store_true')
    p.add_argument('--box-orientation-feedback',action='store_true')
    p.add_argument('--track-source-rotation',action='store_true')
    p.add_argument('--native-collision-contacts',action='store_true')
    a=p.parse_args();run(a.out.resolve(),a.plan.resolve(),a.model.resolve(),a.acceleration_limit,a.velocity_limit,a.torso_filter_seconds,a.force_grasp,a.joint_limit_cost,a.soft_joint_margin,a.collision_braking,a.track_source,a.lead_steps,a.orientation_feedback,a.grasp_travel,a.track_source_rotation,a.synchronized_grasp,a.box_orientation_feedback,a.native_collision_contacts,a.grasp_height_adjustment)
