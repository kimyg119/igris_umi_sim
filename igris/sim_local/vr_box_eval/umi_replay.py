"""UMI relative-pose audit and guarded native SDK/MuJoCo replay; no dataset edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import evaluate
from igris_ik import IKConfig, UpperBodyIK
from igris_ik.assets import HAND_TOOL_FRAMES
from igris_ik.configuration import RobotConfiguration
from igris_ik.constants import URDF_JOINT_NAMES, UPPER_BODY_FIXED_SDK_INDICES
from igris_ik.geometry import Pose, UpperBodyTargets
from igris_ik.realtime.cartesian import track_step
from igris_ik.realtime.robot_control import ControlConfig, ARM_GAIN_INIT
from igris_ik.realtime.retarget import hmd_waist_yaw
from render_results import model_for


def load_relative(root):
    meta = json.loads((root/'meta/relative_pose.json').read_text())
    if meta['representation'] != 'per_hand_episode_start_se3' or meta['position_unit'] != 'metre':
        raise ValueError('Expected episode-start relative SE(3), metres')
    with np.load(root/'relative_targets.npz', allow_pickle=False) as source:
        times = source['host_elapsed_s'].copy()
        frames = source['frame_index'].copy()
        poses = np.stack([source[s+'_ee_xyzw'] for s in ('left','right')], axis=1).astype(float)
        episodes = source['episode_index'].copy()
    if len(np.unique(episodes)) != 1 or len(times)<2 or poses.shape!=(len(times),2,7):
        raise ValueError('Expected one episode with two (N,7) pose arrays')
    if not np.isfinite(poses).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Nonfinite poses or non-increasing source times')
    if not np.allclose(np.linalg.norm(poses[:,:,3:],axis=-1),1,atol=1e-3):
        raise ValueError('Invalid xyzw quaternions')
    if not np.allclose(poses[0], [[0,0,0,0,0,0,1]]*2, atol=1e-6):
        raise ValueError('First pose must be identity for each hand')
    return times-times[0], frames, poses


def reanchor(relative, start):
    """B_T_E(t) = B_T_E(0) D(t), explicitly assuming tracker-to-EE X=I."""
    result = np.empty_like(relative, dtype=float)
    for i, side in enumerate(('left','right')):
        pose = getattr(start,side)
        rotation = Rotation.from_quat(pose.wxyz[[1,2,3,0]])
        result[:,i,:3] = pose.position + rotation.apply(relative[:,i,:3])
        result[:,i,3:] = (rotation*Rotation.from_quat(relative[:,i,3:])).as_quat()
    return result


def target_at(poses, torso):
    return UpperBodyTargets(**{side:Pose(poses[i,:3],poses[i,[6,3,4,5]])
                               for i,side in enumerate(('left','right'))},torso=torso)


def inferred_yaw(poses):
    """Virtual head follows the hand-pair bearing; ambiguous spans under 5 cm hold yaw."""
    pairs=poses[:,1,:2]-poses[:,0,:2]
    zero=np.arctan2(pairs[0,1],pairs[0,0]);previous=0.;result=[]
    for pair in pairs:
        if np.linalg.norm(pair)>=.05:
            bearing=np.arctan2(pair[1],pair[0])
            previous=(bearing-zero+np.pi)%(2*np.pi)-np.pi
        result.append(previous)
    return np.asarray(result)


def pose_array(target):
    return np.array([np.r_[getattr(target,s).position,getattr(target,s).wxyz[[1,2,3,0]]]
                     for s in ('left','right')])


def pose_error(actual, desired):
    position = np.linalg.norm(actual[...,:3]-desired[...,:3],axis=-1)*1000
    dot = np.abs(np.sum(actual[...,3:]*desired[...,3:],axis=-1))
    angle = np.rad2deg(2*np.arccos(np.clip(dot,0,1)))
    return position, angle


def config():
    # Same controller-only cost and hard gates as the VR path; no human arm tracking.
    return IKConfig(null_space_weight={'waist_yaw':.1,'waist_pitch':.2,'waist_roll':.2,
                     'arm':.016,'neck':.05},local_acceleration_weight=.001,local_num_iters=40,
                     local_joint_limit_weight=100000.,self_collision_cost_margin_m=.001,
                     torso_axes_weight=(0,0,0,5,5,5),elbow_out_weight=.2,recovery_after_frames=0)


def prepare_scene(out):
    old = evaluate.READY_JOINTS.copy()
    forward = {f'{s}_{joint}':value for s in ('l','r') for joint,value in
               [('shoulder_pitch',-1.37),('shoulder_roll',.12 if s=='l' else -.12),
                ('shoulder_yaw',0.),('elbow_pitch',-.4),('wrist_yaw',0.),('wrist_roll',0.),('wrist_pitch',0.)]}
    try:
        evaluate.READY_JOINTS.clear();evaluate.READY_JOINTS.update(forward)
        evaluate.scene(.01,out)
    finally:
        evaluate.READY_JOINTS.clear();evaluate.READY_JOINTS.update(old)
    tree = ET.parse(out/'scene.xml');root=tree.getroot();world=root.find('worldbody')
    for child in list(world):
        if child.get('name','').startswith(('eval_box','eval_table')):world.remove(child)
    for key in root.findall('./keyframe/key'):
        key.set('qpos',' '.join(key.get('qpos').split()[:-7]))  # Removed box free joint only.
    tree.write(out/'scene.xml',encoding='unicode')
    model=model_for(out)
    assert model.joint('floating_base').type == mujoco.mjtJoint.mjJNT_FREE
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'eval_box') == -1
    return forward


def native(root, out, infer_yaw=False):
    from igris_ik.gui.robot_monitor import RobotMonitor
    from igris_ik.realtime.robot_io import RobotCommandPublisher
    times,frames,relative=load_relative(root)
    forward=prepare_scene(out)
    initial=json.loads((out/'motions/HOME.json').read_text())['motions'][0]['position']
    ik=UpperBodyIK(config());ik.warmup(initial);ik.lock_to(initial)
    check=ik.check_configuration(initial,ik.forward(initial))
    if not check.success:raise RuntimeError(f'Forward start pose invalid: {check.reasons}')
    evaluate.setup(out)
    control=ControlConfig(slew_rad_s=.5,accel_rad_s2=2.,arm_gains=ARM_GAIN_INIT,
                          hold_untracked=False,command_waist=True,required_robot_mode='ROBOT_STATE_WALK_LOW',
                          reference_timeout_s=.3,ik_orientation_gate_rad=np.deg2rad(20),ik_residual_gate_frames=1)
    monitor=RobotMonitor(ik,'tcp://172.17.0.2:5560','tcp://172.17.0.2:5561',port=8093,
                         control=control,log_dir=out,gravity_align=False,branch_guard_rad=.1,
                         allow_manual_follow=False)
    commands=[];send=RobotCommandPublisher.send
    def observe(publisher,command):
        commands.append([time.time(),*command.q]);return send(publisher,command)
    RobotCommandPublisher.send=observe
    rows=[]
    try:
        monitor.start(wait_for_state_s=3.)
        state=monitor.subscriber.last_state
        if state.mode!='ROBOT_STATE_WALK_LOW':raise RuntimeError(state.mode)
        error=max(abs(state.q[URDF_JOINT_NAMES.index(n)]-v) for n,v in forward.items())
        if error>.35:raise RuntimeError(f'Forward HOME error: {error} rad')
        ik.lock_to(state.q);monitor.arm()
        q0=monitor.controller.q_cmd.copy();start=ik.forward(q0);targets=reanchor(relative,start)
        yaw=inferred_yaw(targets) if infer_yaw else np.zeros(len(targets))
        torso_goals=[];waist_goals=[]
        for angle in yaw:
            anchor=q0.copy()
            head=Pose([0,0,0],Rotation.from_euler('z',angle).as_quat()[[3,0,1,2]])
            anchor[0]=hmd_waist_yaw(head,0.,q0[0],np.deg2rad(ik.config.waist_limits_deg['waist_yaw']))
            torso_goals.append(ik.forward(anchor).torso);waist_goals.append(anchor[0])
        origin=monitor.begin_cartesian(RobotConfiguration.from_sdk(q0))
        if not origin.success:raise RuntimeError(f'Cannot arm forward pose: {origin.reasons}')
        accepted=UpperBodyTargets(start.left,start.right,torso=start.torso)
        metadata=dict(dataset=str(root),input_sha256=hashlib.sha256((root/'relative_targets.npz').read_bytes()).hexdigest(),
                      mount_assumption='X=I; each initial tracker axis aligned with corresponding initial URDF end-effector axis; zero offset',
                      target_frame='robot_base / pelvis; follows free base, no world-odometry compensation',
                      tools=list(HAND_TOOL_FRAMES),scale=1.,arm_tracking='off',
                      torso=('virtual HMD yaw from initial-relative horizontal hand-pair bearing; roll/pitch held; span<5cm holds yaw' if infer_yaw else 'initial orientation target; no HMD recorded'),
                      inferred_hmd=infer_yaw,initial_q=q0.tolist(),
                      source_frames=len(frames),source_duration_s=float(times[-1]),first_pose_error_rad=error,
                      cartesian_speed_m_s=.15,cartesian_angular_speed_rad_s=.6,
                      wrist_kp=30.,wrist_kd=3.,free_base=True,box=False,supports=False,
                      execution='all original rows; each step takes at least its original source interval; no frame skipping',
                      start_wall=time.time())
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        np.savez_compressed(out/'targets.npz',source_t=times,frame_index=frames,poses_xyzw=targets,
                            torso_position=np.array([p.position for p in torso_goals]),
                            torso_wxyz=np.array([p.wxyz for p in torso_goals]),
                            inferred_hmd_yaw_rad=yaw,waist_goal_rad=np.array(waist_goals))
        (out/'record.trigger').write_text(str(metadata['start_wall']))
        monitor.set_reference_active(True);monitor.follow();locked=q0.copy()
        with (out/'native_frames.jsonl').open('w',buffering=1) as log:
            for index,t in enumerate(times):
                began=time.time();state=monitor.subscriber.last_state
                if state is None or began-state.t_wall>.3:raise RuntimeError('Stale vendor state')
                if np.max(abs(state.q[UPPER_BODY_FIXED_SDK_INDICES]-locked[UPPER_BODY_FIXED_SDK_INDICES]))>.01:
                    ik.lock_to(state.q,preserve_warm_start=True);locked=state.q.copy()
                goal=target_at(targets[index],torso_goals[index]);dt=float(times[index]-times[index-1]) if index else 1/30
                monitor.set_reference_active(True)
                result,waypoint,fraction=track_step(monitor,accepted,goal,dt)
                if result.success:accepted=waypoint
                candidate=pose_array(ik.forward(result.q_sdk))
                raw_error=pose_error(candidate,targets[index])
                status=monitor.snapshot()
                row=dict(frame_index=int(frames[index]),source_t=float(t),wall_time=time.time(),
                         success=result.success,reasons=list(result.reasons),fraction=float(fraction),
                         candidate_q=result.q_sdk.tolist(),candidate_fk_xyzw=candidate.tolist(),
                         candidate_raw_position_error_mm=raw_error[0].tolist(),candidate_raw_orientation_error_deg=raw_error[1].tolist(),
                         accepted_target_xyzw=pose_array(accepted).tolist(),controller_mode=status['mode'],trip=status['last_trip'])
                log.write(json.dumps(row)+'\n');rows.append(row)
                if index%150==0:print(f'REPLAY {index}/{len(times)} source={t:.1f}s mode={status["mode"]} accepted={sum(r["success"] for r in rows)}',flush=True)
                # Keep a real safety trip latched; continue auditing raw inputs without rearming.
                time.sleep(max(0.,dt-(time.time()-began)))
        monitor.set_reference_active(False);monitor.stop('replay_complete');time.sleep(.5)
        metadata['duration_s']=time.time()-metadata['start_wall']
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    finally:
        monitor.set_reference_active(False);monitor.stop('replay_end');monitor.close()
        RobotCommandPublisher.send=send
        np.save(out/'commands.npy',np.asarray(commands))
        evaluate.ctl('sim','pause')
    print('NATIVE COMPLETE',out,flush=True)


def audit(out):
    """Every original target, global static IK. No branch/timing success implied."""
    meta=json.loads((out/'metadata.json').read_text());data=np.load(out/'targets.npz');q=np.array(meta['initial_q'])
    ik=UpperBodyIK(config());ik.warmup(q);ik.lock_to(q)
    torso_p=np.broadcast_to(data['torso_position'],(len(data['source_t']),3))
    torso_q=np.broadcast_to(data['torso_wxyz'],(len(data['source_t']),4))
    rows=[]
    with (out/'static_ik.jsonl').open('w',buffering=1) as log:
        for i,pose in enumerate(data['poses_xyzw']):
            goal=target_at(pose,Pose(torso_p[i],torso_q[i]));ik.reset();result=ik.solve(q,goal)
            actual=pose_array(ik.forward(result.q_sdk));position,angle=pose_error(actual,pose)
            row=dict(frame_index=int(data['frame_index'][i]),source_t=float(data['source_t'][i]),success=result.success,
                     feasible=result.feasible,collision_free=result.collision_free,reasons=list(result.reasons),
                     position_error_mm=position.tolist(),orientation_error_deg=angle.tolist(),q=result.q_sdk.tolist(),
                     solve_ms=result.diagnostics.solve_time_s*1000)
            log.write(json.dumps(row)+'\n');rows.append(row)
            if i%200==0:print(f'STATIC IK {i}/{len(data["poses_xyzw"])} accepted={sum(r["success"] for r in rows)}',flush=True)
    print('AUDIT COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('path',type=Path)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--infer-yaw',action='store_true');args=parser.parse_args()
    if args.audit:audit(args.path.resolve())
    else:
        out=evaluate.HERE/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_umi_relative{"_yaw" if args.infer_yaw else ""}'
        out.mkdir(exist_ok=False)
        try:native(args.path.resolve(),out,args.infer_yaw)
        finally:evaluate.ctl('sim','pause')
        audit(out)
