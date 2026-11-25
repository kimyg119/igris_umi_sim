"""Saved Quest poses through bounded Cartesian IK and the native simulation SDK.

This is an ordered file trajectory, not a replay of missing live grip events.
Original targets remain in the logs; only existing reach/MoveL guards limit commands.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import evaluate
from igris_ik import IKConfig
from igris_ik.assets import HAND_TOOL_FRAMES, FIXED_HAND_JOINT_VALUES, hand_tool_transforms
from igris_ik.configuration import RobotConfiguration
from igris_ik.constants import URDF_JOINT_NAMES
from igris_ik.geometry import UpperBodyTargets
from igris_ik.gui.robot_monitor import RobotMonitor
from igris_ik.realtime.cartesian import track_step
from igris_ik.realtime.retarget import DeltaCalibration, clamp_bimanual_reach, hmd_waist_yaw, SHOULDER_LEFT, SHOULDER_RIGHT, REACH_M
from igris_ik.realtime.robot_control import ControlConfig, ARM_GAIN_INIT, ARM_GROUPS
from igris_ik.realtime.separated_waist import SeparatedWaistIK
from igris_ik.realtime.zmq_source import parse_pose_message
from render_results import model_for, pose_axes
from render_umi_replay import stats
from umi_replay import prepare_scene, pose_array, pose_error


def load_capture(path):
    payloads=[json.loads(line) for line in path.open() if line.strip()]
    messages=[parse_pose_message(json.dumps(row)) for row in payloads]
    if len(messages)<2 or any(m.hmd is None for m in messages):
        raise ValueError('Expected at least two frames with head pose')
    ns=[m.t_capture_monotonic_ns for m in messages]
    if any(n is None for n in ns):
        raise ValueError('Monotonic capture times are required')
    times=np.array([n-ns[0] for n in ns],dtype=float)*1e-9
    if np.any(np.diff(times)<=0) or np.any(np.diff([m.seq for m in messages])<=0):
        raise ValueError('Capture times and sequences must strictly increase')
    gaps=np.flatnonzero(np.diff(times)>.3)+1
    jumps={}
    for name in ('left','right','hmd'):
        poses=[getattr(m,name) for m in messages]
        p=np.array([x.position for x in poses]);r=Rotation.from_quat([x.wxyz[[1,2,3,0]] for x in poses])
        distance=np.linalg.norm(np.diff(p,axis=0),axis=1);angle=np.rad2deg((r[:-1].inv()*r[1:]).magnitude())
        jumps[name]=[dict(index=int(i),source_s=float(times[i]),step_mm=float(distance[i-1]*1000),step_deg=float(angle[i-1]))
                     for i in np.flatnonzero((distance>.1)|(angle>30))+1]
    audit=dict(frames=len(messages),duration_s=float(times[-1]),quaternion='xyzw',frame='robot_base',
               missing_buttons=sum('buttons' not in r for r in payloads),
               constant_seed=len({json.dumps(r.get('seed_qpos'),sort_keys=True) for r in payloads})==1,
               gaps=[dict(index=int(i),source_s=float(times[i]),gap_s=float(times[i]-times[i-1])) for i in gaps],jumps=jumps)
    return messages,times,audit


def make_targets(messages, ik, q0, mapping='absolute'):
    source=lambda m:UpperBodyTargets(left=m.left,right=m.right,head=m.hmd)
    calibration=DeltaCalibration(delta_scale=1.,body_relative=False)
    calibration.calibrate(source(messages[0]),ik.forward(q0))
    targets=[];waist=[]
    for m in messages:
        mapped=calibration.map(source(m)) if mapping=='clutch' else source(m)
        anchor=q0.copy()
        anchor[0]=hmd_waist_yaw(m.hmd,calibration.head_yaw_rad if mapping=='clutch' else 0.,
            q0[0] if mapping=='clutch' else 0.,np.deg2rad(ik.config.waist_limits_deg['waist_yaw']))
        targets.append(UpperBodyTargets(mapped.left,mapped.right,torso=ik.forward(anchor).torso));waist.append(anchor[0])
    return targets,waist


def ik_config():
    # Match the current controller-only VR cost; use its existing separated-waist solver.
    return IKConfig(waist_mode='fixed',neck_mode='fixed',third_keypoint='torso',
        null_space_weight={'waist_yaw':.1,'waist_pitch':.2,'waist_roll':.2,'arm':.016,'neck':.05},
        rest_weight=5./64,local_acceleration_weight=.001/64,local_num_iters=40,
        local_joint_limit_weight=100000.,self_collision_cost_margin_m=.001,
        torso_axes_weight=(0,0,0,2.5,2.5,.625),torso_yaw_only=True,
        elbow_position_weight=0.,elbow_out_weight=.05,elbow_out_offset_m=.06,
        hand_axes_weight=(1.,1.,1.,.5,.5,.5),recovery_after_frames=0)


def refresh_file_reference(monitor, stopped):
    """A fully loaded file stays available while IK works; do not use for live VR."""
    while not stopped.wait(.02):
        monitor.set_reference_active(True)


def native(capture, out, profile, mapping):
    messages,times,audit=load_capture(capture)
    gains={g:(400.,5.) for g in ARM_GROUPS} if profile=='400_5' else {g:(kp,min(kd,5.)) for g,(kp,kd) in ARM_GAIN_INIT.items()}
    out.mkdir(parents=True)
    shutil.copy2(capture,out/'source.jsonl');shutil.copy2(__file__,out/Path(__file__).name)
    (out/'source_audit.json').write_text(json.dumps(audit,indent=2))
    forward=prepare_scene(out)
    (out/'record.trigger').write_text('preflight')
    evaluate.setup(out,sim_arm_gains=gains)
    config=ik_config()
    ik=SeparatedWaistIK(config);ik.warmup(ik.default_configuration())
    control=ControlConfig(slew_rad_s=.5,accel_rad_s2=2.,arm_gains=gains,hold_untracked=False,
        command_waist=True,required_robot_mode='ROBOT_STATE_WALK_LOW',reference_timeout_s=.3,
        ik_orientation_gate_rad=np.deg2rad(20),ik_residual_gate_frames=1)
    monitor=RobotMonitor(ik,'tcp://172.17.0.2:5560','tcp://172.17.0.2:5561',port=8093,
        control=control,log_dir=out,gravity_align=False,branch_guard_rad=.1,allow_manual_follow=False)
    metadata={};reference_stop=threading.Event();reference_thread=None
    try:
        monitor.start(wait_for_state_s=3.);monitor.set_recording(True)
        state=monitor.subscriber.last_state
        if state is None or state.mode!='ROBOT_STATE_WALK_LOW':raise RuntimeError('WALK_LOW unavailable')
        ready_error=max(abs(state.q[URDF_JOINT_NAMES.index(n)]-v) for n,v in forward.items())
        if ready_error>.35:raise RuntimeError(f'Forward HOME error {ready_error} rad')
        ik.lock_to(state.q,joint_offsets=state.joint_offsets);monitor.arm()
        q0=monitor.controller.q_cmd.copy();ik.set_waist_reference(q0);start=ik.forward(q0)
        origin=monitor.begin_cartesian(RobotConfiguration.from_sdk(q0))
        if not origin.success:raise RuntimeError(f'Initial hold invalid: {origin.reasons}')
        targets,waist=make_targets(messages,ik,q0,mapping)
        np.savez_compressed(out/'targets.npz',source_t=times,seq=[m.seq for m in messages],
            poses_xyzw=np.array([pose_array(t) for t in targets]),waist_goal_rad=waist)
        metadata=dict(profile=profile,sim_arm_gains=gains,source_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
            source_frames=len(messages),source_duration_s=float(times[-1]),mapping=mapping,
            waist_mapping='clutched HMD yaw' if mapping=='clutch' else 'absolute pelvis HMD yaw, SDK sign and joint limits',
            seed_qpos_used=False,synthetic_enable='ordered file execution; no original grips recorded',
            reference_liveness='Fully loaded file, independently refreshed every 20ms; reference_timeout_s remains 0.3; no automatic rearm.',
            frame='pelvis; no world odometry compensation',scale=1.,arm_tracking='off',separate_waist=True,
            free_base=True,box=False,initial_q=q0.tolist(),first_pose_error_rad=ready_error,
            fixed_hand_joint_values=FIXED_HAND_JOINT_VALUES,tools=list(HAND_TOOL_FRAMES),
            ik_config=repr(config),execution='Every source target once; each interval is at least the original next-source interval. Existing MoveL/reach/IK/SDK guards. Source gaps are held, not live packet timeouts.',start_wall=time.time())
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
        (out/'record.trigger').write_text(str(metadata['start_wall']))
        defaults=ik.forward(ik.default_configuration());wrist_offsets=np.array([-t[:3,:3].T@t[:3,3] for t in hand_tool_transforms()])
        accepted=UpperBodyTargets(start.left,start.right,torso=start.torso)
        monitor.set_reference_active(True);monitor.follow()
        reference_thread=threading.Thread(target=refresh_file_reference,args=(monitor,reference_stop),daemon=True)
        reference_thread.start()
        def step(goal, waist_goal, dt):
            nonlocal accepted
            state=monitor.subscriber.last_state
            if state is None or not 0<=time.time()-state.t_wall<.3:raise RuntimeError('Stale SDK state')
            if state.joint_offsets is None or not 0<=time.time()-state.joint_offsets_t_wall<.15:raise RuntimeError('Stale backlash observation')
            ik.lock_to(state.q,preserve_warm_start=True,joint_offsets=state.joint_offsets)
            reach_q=state.q.copy();reach_q[0]=waist_goal;torso=ik.forward(reach_q).torso
            rot=Rotation.from_quat(torso.wxyz[[1,2,3,0]])*Rotation.from_quat(defaults.torso.wxyz[[1,2,3,0]]).inv()
            shoulders=[torso.position+rot.apply(p-defaults.torso.position) for p in (SHOULDER_LEFT,SHOULDER_RIGHT)]
            left,right,hit=clamp_bimanual_reach(goal.left,goal.right,*shoulders,REACH_M,preserve_relative=False,wrist_offsets=wrist_offsets)
            limited=UpperBodyTargets(left,right,torso=goal.torso)
            monitor.set_reference_active(True)
            result,waypoint,fraction=track_step(monitor,accepted,limited,dt)
            hold_failure=None
            if result.success:accepted=waypoint
            else:
                held=monitor.check_cartesian_hold()
                if held.feasible:
                    fk=ik.forward(held.configuration);accepted=UpperBodyTargets(fk.left,fk.right,torso=fk.torso)
                else:
                    hold_failure=dict(reasons=list(held.reasons),q_sdk=held.q_sdk.tolist(),joint_offsets=state.joint_offsets.tolist())
                    monitor.stop('held_pose_invalid')
            status=monitor.snapshot()
            return dict(wall_time=time.time(),success=result.success,
                reasons=list(result.reasons),collision_free=result.collision_free,feasible=result.feasible,
                fraction=float(fraction),reach_clamped=bool(hit),accepted_xyzw=pose_array(accepted).tolist(),
                limited_xyzw=pose_array(limited).tolist(),candidate_xyzw=pose_array(ik.forward(result.q_sdk)).tolist(),
                candidate_q=result.q_sdk.tolist(),solve_ms=1000*result.diagnostics.solve_time_s,
                ik_waypoint_error_m=result.diagnostics.max_hand_position_error_m,
                mode=status['mode'],trip=status['last_trip'],hold_failure=hold_failure)

        if mapping=='absolute':
            metadata['entry_start_wall']=time.time()
            with (out/'entry.jsonl').open('w',buffering=1) as log:
                for j in range(300):
                    began=time.monotonic();row=step(targets[0],waist[0],1/30)
                    pe,ae=pose_error(pose_array(ik.forward(monitor.subscriber.last_state.q)),pose_array(targets[0]))
                    reached=bool(np.all(pe<15)&np.all(ae<20))
                    row.update(index=j,position_mm=pe.tolist(),orientation_deg=ae.tolist(),reached=reached)
                    log.write(json.dumps(row)+'\n')
                    if j%100==0:print(f'{profile} ENTRY {j}: {pe.round(1)} mm {row["reasons"]}',flush=True)
                    if reached or row['mode']!='FOLLOWING':break
                    time.sleep(max(0.,1/30-(time.monotonic()-began)))
            metadata['entry']=dict(steps=j+1,reached=reached,position_mm=pe.tolist(),orientation_deg=ae.tolist(),
                duration_s=time.time()-metadata['entry_start_wall'],trip=row['trip'])
            # Keep the bounded HOME transition separate from the source clock and metrics.
            metadata['start_wall']=time.time()
            (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        with (out/'steps.jsonl').open('w',buffering=1) as log:
            for i,goal in enumerate(targets):
                began=time.monotonic()
                row=step(goal,waist[i],float(times[i]-times[i-1]) if i else 1/30)
                row.update(index=i,seq=messages[i].seq,source_t=float(times[i]))
                log.write(json.dumps(row)+'\n')
                if i%100==0:print(f'{profile} {i}/{len(targets)} {row["mode"]} {row["reasons"]}',flush=True)
                # A known file trajectory stays alive during recorded sampling gaps.
                # Never rearm a latched controller fault.
                interval=float(times[i+1]-times[i]) if i+1<len(times) else 0.
                while time.monotonic()-began<interval:
                    monitor.set_reference_active(True);time.sleep(min(.02,max(0.,interval-(time.monotonic()-began))))
        hold=time.monotonic()
        while time.monotonic()-hold<2.:
            monitor.set_reference_active(True);time.sleep(.02)
        metadata['duration_s']=time.time()-metadata['start_wall']
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    finally:
        reference_stop.set()
        if reference_thread is not None:reference_thread.join()
        monitor.set_reference_active(False);monitor.stop('replay_end');monitor.close();evaluate.ctl('sim','pause')
    print('REPLAY COMPLETE',out,flush=True)


def render(out):
    """Measure and render recorded native qpos; do not step host physics."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    meta=json.loads((out/'metadata.json').read_text());target=np.load(out/'targets.npz')
    messages,_,_=load_capture(out/'source.jsonl')
    head_yaw=Rotation.from_quat([m.hmd.wxyz[[1,2,3,0]] for m in messages]).as_euler('xyz')[:,2]
    steps=[json.loads(line) for line in (out/'steps.jsonl').open()]
    assert len(steps)==meta['source_frames'] and [x['index'] for x in steps]==list(range(len(steps)))
    raw=(out/'physics.bin').read_bytes();nq,nv=struct.unpack('ii',raw[:8]);width=2+nq+nv+7
    assert (len(raw)-8)%(8*width)==0
    records=np.frombuffer(raw[8:],np.float64).reshape(-1,width)
    t=records[:,0]-meta['start_wall'];keep=(t>=0)&(t<=meta['duration_s']);records=records[keep];t=t[keep]
    model=model_for(out);assert (model.nq,model.nv)==(nq,nv) and model.joint('floating_base').type==mujoco.mjtJoint.mjJNT_FREE
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'eval_box')==-1
    data=mujoco.MjData(model);base=model.body('pelvis').id;hands=[model.body(s).id for s in meta['tools']]
    actual=[];positions=[];tilts=[];palm_angles=[]
    hand_roots=[model.body(s+'_hand_rcv').id for s in ('l','r')]
    for row in records:
        data.qpos[:]=row[2:2+nq];mujoco.mj_forward(model,data)
        bp=data.xpos[base];br=Rotation.from_matrix(data.xmat[base].reshape(3,3))
        actual.append([np.r_[br.inv().apply(data.xpos[b]-bp),(br.inv()*Rotation.from_matrix(data.xmat[b].reshape(3,3))).as_quat()] for b in hands])
        positions.append(bp.copy());tilts.append(np.rad2deg(np.arccos(np.clip(br.as_matrix()[2,2],-1,1))))
        rotations=np.array([data.xmat[b].reshape(3,3) for b in hand_roots]);normal=rotations[:,:,0];finger=rotations[:,:,2]
        direction=data.xpos[hands[1]]-data.xpos[hands[0]];direction=direction/max(np.linalg.norm(direction),1e-12)
        palm_angles.append(np.rad2deg(np.arccos(np.clip([normal[0]@direction,-normal[1]@direction,-normal[0]@normal[1],finger[0]@finger[1]],-1,1))))
    actual=np.array(actual);positions=np.array(positions)
    stamps=np.array([r['wall_time'] for r in steps]);indices=np.clip(np.searchsorted(records[:,0],stamps,side='right')-1,0,len(records)-1)
    desired=target['poses_xyzw'];accepted=np.array([r['accepted_xyzw'] for r in steps]);candidate=np.array([r['candidate_xyzw'] for r in steps])
    pos,ang=pose_error(actual[indices],desired);ap,aa=pose_error(actual[indices],accepted);cp,ca=pose_error(candidate,desired)
    success=np.array([r['success'] for r in steps]);following=np.array([r['mode']=='FOLLOWING' for r in steps])
    waypoint_p,waypoint_a=pose_error(candidate[success],accepted[success])
    limited=np.array([r['limited_xyzw'] for r in steps])
    measured_yaw=records[indices,2+int(model.joint('waist_yaw').qposadr[0])]
    summary=dict(profile=meta['profile'],mapping=meta['mapping'],source_frames=len(steps),source_duration_s=meta['source_duration_s'],replay_duration_s=meta['duration_s'],
        entry=meta.get('entry'),first_trip=next(({k:r.get(k) for k in ('index','source_t','wall_time','trip','hold_failure')} for r in steps if r['trip']),None),
        accepted_steps=int(success.sum()),following_steps=int(following.sum()),reach_clamped_steps=sum(r['reach_clamped'] for r in steps),
        active_accepted_steps=int(np.sum(success&following)),following_raw_position_mm=stats(pos[following]),
        following_accepted_position_mm=stats(ap[following]),
        palm_alignment_metric_names=['left_faces_right','right_faces_left','palm_opposition','finger_parallel'],
        following_palm_alignment_deg=stats(np.array(palm_angles)[indices][following]),
        following_palms_within_15deg_fraction=float(np.mean(np.all(np.array(palm_angles)[indices][following]<15,axis=1))) if following.any() else None,
        full_target_steps=sum(r['success'] and r['fraction']==1 and not r['reach_clamped'] for r in steps),
        reasons=dict(Counter(x for r in steps if not r['success'] for x in r['reasons'])),trips=dict(Counter(r['trip'] for r in steps if r['trip'])),
        raw_target_position_mm=stats(pos),raw_target_orientation_deg=stats(ang),
        accepted_target_position_mm=stats(ap),accepted_target_orientation_deg=stats(aa),
        successful_ik_waypoint_position_mm=stats(waypoint_p),successful_ik_waypoint_orientation_deg=stats(waypoint_a),
        reach_correction_mm=stats(1000*np.linalg.norm(desired[:,:,:3]-limited[:,:,:3],axis=2)),
        waist_goal_range_deg=np.rad2deg([target['waist_goal_rad'].min(),target['waist_goal_rad'].max()]).tolist(),
        waist_measured_range_deg=np.rad2deg([measured_yaw.min(),measured_yaw.max()]).tolist(),
        waist_requested_error_deg=stats(np.rad2deg(abs(measured_yaw-target['waist_goal_rad']))),
        candidate_raw_position_mm=stats(cp),candidate_raw_orientation_deg=stats(ca),
        ik_waypoint_position_mm=stats([1000*r['ik_waypoint_error_m'] for r in steps]),solve_ms=stats([r['solve_ms'] for r in steps]),
        raw_within_10mm_20deg_fraction=float(np.mean(np.all((pos<=10)&(ang<=20),axis=1))),
        base_tilt_max_deg=float(max(tilts)),base_min_z_m=float(positions[:,2].min()),
        base_xy_max_displacement_m=float(np.linalg.norm(positions[:,:2]-positions[0,:2],axis=1).max()),
        fallen=bool(positions[:,2].min()<.5 or max(tilts)>30),physical_self_contact_frames=int(np.sum(records[:,-4]>0)),
        physics_realtime_ratio=float((records[-1,1]-records[0,1])/(t[-1]-t[0])),max_record_gap_s=float(np.diff(t).max()),
        measurement='Native FK in measured pelvis frame, at each completed IK step timestamp; raw target and accepted waypoint reported separately.')
    sdk=list(csv.DictReader(next(out.glob('20*.csv')).open()))
    sdk=[r for r in sdk if meta['start_wall']<=float(r['t_wall'])<=meta['start_wall']+meta['duration_s']]
    summary['sdk_gains_and_motor_error']={}
    for group,axes in ARM_GROUPS.items():
        vec=lambda prefix:np.array([[float(r[f'{prefix}_{i}']) for i in axes] for r in sdk])
        kp,kd=vec('kp'),vec('kd');assert np.allclose(kp,meta['sim_arm_gains'][group][0]) and np.allclose(kd,meta['sim_arm_gains'][group][1])
        error=np.rad2deg(vec('cmd_q')-vec('q'))
        summary['sdk_gains_and_motor_error'][group]=dict(kp=[float(kp.min()),float(kp.max())],kd=[float(kd.min()),float(kd.max())],rms_deg=float(np.sqrt(np.mean(error**2))),p95_deg=float(np.percentile(abs(error),95)))
        active=error[[r['mode']=='FOLLOWING' for r in sdk]]
        summary['sdk_gains_and_motor_error'][group].update(following_rms_deg=float(np.sqrt(np.mean(active**2))),following_samples=len(active))
    effort_raw=(out/'physics.bin.effort').read_bytes();nu,env=struct.unpack('ii',effort_raw[:8]);assert env==nv
    effort=np.frombuffer(effort_raw[8:],np.float64).reshape(-1,2+2*nu+nv)
    effort=effort[(effort[:,0]>=meta['start_wall'])&(effort[:,0]<=meta['start_wall']+meta['duration_s'])]
    summary['native_motor_torque']={}
    for group,axes in ARM_GROUPS.items():
        axes=np.array(axes);limit=model.actuator_ctrlrange[axes,1]
        force=effort[:,2+nu+axes];ctrl=effort[:,2+axes]
        assert np.max(abs(effort[:,2+nu+axes+31]+effort[:,2+nu+axes+62]))<1e-8
        summary['native_motor_torque'][group]=dict(max_Nm=float(abs(force).max()),p95_Nm=float(np.percentile(abs(force),95)),saturation_fraction=float(np.mean(abs(ctrl)>=.99*limit)))
    with (out/'per_target_errors.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['index','seq','source_s','replay_s','accepted','following','reach_clamped','left_raw_mm','right_raw_mm','left_raw_deg','right_raw_deg','left_accepted_mm','right_accepted_mm','ik_ms','reasons'])
        for i,r in enumerate(steps):writer.writerow([i,r['seq'],r['source_t'],r['wall_time']-meta['start_wall'],r['success'],r['mode'],r['reach_clamped'],*pos[i],*ang[i],*ap[i],r['solve_ms'],';'.join(r['reasons'])])
    video_end=min(t[-1],summary['first_trip']['wall_time']-meta['start_wall']+2.) if summary['first_trip'] else t[-1]
    summary['video_duration_s']=float(video_end)
    summary['video_scope']='Main replay through first latched stop plus 2 seconds, or entire replay if no stop. Entry excluded.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'physics.npz',t=t,qpos=records[:,2:2+nq],qvel=records[:,2+nq:2+nq+nv])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for side,label in enumerate(('left','right')):
        axes[0].plot(target['source_t'],pos[:,side],label=label+' raw');axes[0].plot(target['source_t'],ap[:,side],'--',label=label+' accepted')
        axes[1].plot(target['source_t'],ang[:,side],label=label)
    axes[0].set_ylabel('EE position error mm');axes[1].set_ylabel('EE orientation error deg');axes[1].set_xlabel('Original source time s')
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    fig.suptitle(meta['profile']+' | recorded native FK');fig.tight_layout();fig.savefig(out/'tracking_errors.png');plt.close(fig)
    renderer=mujoco.Renderer(model,720,1280);camera=mujoco.MjvCamera();camera.lookat[:]=[.15,0,1.05];camera.distance=2.1;camera.azimuth=145;camera.elevation=-15
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','24','-i','-','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(out/'replay_targets.mp4')],stdin=subprocess.PIPE)
    step_t=stamps-meta['start_wall']
    for sec in np.arange(0,video_end,1/24):
        k=int(np.clip(np.searchsorted(t,sec,side='right')-1,0,len(t)-1));i=int(np.clip(np.searchsorted(step_t,sec,side='right')-1,0,len(steps)-1))
        data.qpos[:]=records[k,2:2+nq];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);br=Rotation.from_matrix(data.xmat[base].reshape(3,3));bp=data.xpos[base]
        for side in (0,1):
            pose_axes(draw,renderer.scene,bp+br.apply(desired[i,side,:3]),(br*Rotation.from_quat(desired[i,side,3:])).as_matrix())
            pose_axes(draw,renderer.scene,data.xpos[hands[side]],data.xmat[hands[side]].reshape(3,3),actual=True)
        current_pos,current_ang=pose_error(actual[k],desired[i]);current_ap,_=pose_error(actual[k],accepted[i])
        draw.rectangle((0,0,1280,182),fill=(20,25,30));gain='all arms 400 / 5' if meta['profile']=='400_5' else 'shoulder/elbow 121.51 / 5; wrist 30 / 3'
        lines=[f'QUEST saved trajectory | Kp/Kd {gain} | native vendor MuJoCo',
            f'Source {steps[i]["source_t"]:.2f}/{meta["source_duration_s"]:.2f}s | replay {sec:.2f}s | FRAME {i+meta.get("source_frame_offset",0)} | SEQ {steps[i]["seq"]} | {steps[i]["mode"]}',
            f'Raw EE error L/R: {current_pos[0]:.1f}/{current_pos[1]:.1f} mm; {current_ang[0]:.1f}/{current_ang[1]:.1f} deg',
            f'Accepted EE error L/R: {current_ap[0]:.1f}/{current_ap[1]:.1f} mm | reach clamp {steps[i]["reach_clamped"]}',
            f'IK {"ACCEPT" if steps[i]["success"] else "HOLD"}: '+(f'STOP LATCHED: {steps[i]["trip"]}' if steps[i]['trip'] else ', '.join(steps[i]['reasons']) or 'bounded Cartesian waypoint'),
            f'Actual palm opposition / finger parallel errors: {palm_angles[k][2]:.1f} / {palm_angles[k][3]:.1f} deg']
        for n,line in enumerate(lines):draw.text((14,5+29*n),line,font=font,fill='white')
        draw.rectangle((0,650,1280,720),fill=(20,25,30))
        waist_actual=float(data.qpos[int(model.joint('waist_yaw').qposadr[0])])
        draw.text((14,656),f'HMD yaw ({meta.get("hmd_source","input")}): {np.rad2deg(head_yaw[i]):+.1f} deg | waist SDK target / measured: {np.rad2deg(target["waist_goal_rad"][i]):+.1f} / {np.rad2deg(waist_actual):+.1f} deg',font=font,fill='yellow')
        draw.text((14,687),f'Long RGB: requested EE | short pale: native FK | free base WALK_LOW | {meta["mapping"]} mapping, scale 1',font=font,fill='white')
        ff.stdin.write(im.tobytes())
    ff.stdin.close();assert ff.wait()==0;renderer.close();print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--profile',choices=('400_5','vendor_kd5'),default='400_5');p.add_argument('--render',action='store_true')
    p.add_argument('--mapping',choices=('clutch','absolute'),default='absolute');args=p.parse_args()
    if args.render:render(args.capture.resolve())
    else:
        out=evaluate.HERE/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_quest_{args.mapping}_{args.profile}'
        try:native(args.capture.resolve(),out,args.profile,args.mapping)
        finally:evaluate.ctl('sim','pause')
