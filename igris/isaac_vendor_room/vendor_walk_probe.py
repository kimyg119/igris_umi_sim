"""Private empty-hand vendor WALK_LOW waist A/B, using native-time commands.

The native physical model is unchanged. The optional native policy clock and
simulation wall-rate cap isolate scheduler timing. This owns only upper-body
LowCmd and WalkTwist; vendor V5 owns every leg actuator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from run import ROOT,HERE,JOINTS,OWNED

INPUT_FILES=('vendor.xml','manufacturer.xml','scene.xml','robot.yaml','manifest.json','compliance.json','cyclonedds.xml')
GAINS=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml'
DEFAULT_SOURCE=ROOT/'outputs/hlik_transport_20260918_2005/walking_trial59/run'


def twist(elapsed):
    return [0.,0.,min(.23,max(0.,elapsed-3.)*.2)] if 3.<=elapsed<11. else [0.,0.,0.]


def approach(initial,goal):
    initial=np.asarray(initial,float);goal=np.asarray(goal,float).copy()
    goal[3:15]=initial[3:15]
    spread=initial.copy();spread[16]=.9;spread[23]=-.9
    folded=spread.copy();folded[18]=goal[18];folded[25]=goal[25]
    points=np.array([initial,spread,folded,goal])
    times=np.r_[0.,np.cumsum([max(2.,float(np.max(abs(b-a)))*1.875/.3)for a,b in zip(points[:-1],points[1:])])]
    return points,times


def interpolate(points,times,elapsed):
    if elapsed>=times[-1]:return points[-1].copy()
    i=int(np.clip(np.searchsorted(times,elapsed,side='right')-1,0,len(times)-2))
    u=float(np.clip((elapsed-times[i])/(times[i+1]-times[i]),0,1))
    return points[i]+u**3*(10-15*u+6*u*u)*(points[i+1]-points[i])


def require_state(environment,mode,state,age,mode_age):
    if environment!=1 or mode!=6:raise RuntimeError('Fresh SIM WALK_LOW required')
    if not 0<=age<.25 or not 0<=mode_age<.5:raise RuntimeError('Stale SDK31/RobotState')
    if state is None or any(np.asarray(state[k]).shape!=(31,) or not np.isfinite(state[k]).all()for k in ('q','dq','error_code')):
        raise RuntimeError('Finite SDK31 state required')
    if any(state['error_code']):raise RuntimeError('SDK31 joint status error')


def yaw(pose):
    w,x,y,z=pose[3:];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def require_pose(pose,origin):
    p=np.asarray(pose,float);o=np.asarray(origin,float)
    if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>.001:raise RuntimeError('Invalid native pelvis')
    if p[2]<.7:raise RuntimeError('Pelvis height below .7m')
    tilt=math.acos(np.clip(1-2*(p[4]**2+p[5]**2),-1,1))
    if tilt>.3:raise RuntimeError('Pelvis tilt above .3rad')
    heading=yaw(o);lateral=-math.sin(heading)*(p[0]-o[0])+math.cos(heading)*(p[1]-o[1])
    if abs(lateral)>.4:raise RuntimeError('Lateral displacement above .4m')


class UpperLimits:
    def __init__(self):
        tree=ET.parse(ROOT/'igris/igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf')
        limits=[tree.find(f".//joint[@name='{JOINTS[i]}']/limit")for i in OWNED]
        self.lower=np.array([float(x.get('lower'))for x in limits]);self.upper=np.array([float(x.get('upper'))for x in limits])
        sys.path.insert(0,str(ROOT/'igris/igris_curobo/src'))
        from igris_ik.vendor_limits import coupled_halfspaces
        self.a,self.b=coupled_halfspaces([JOINTS[i]for i in OWNED])

    def check(self,q):
        q=np.asarray(q,float)
        if q.shape!=(31,) or not np.isfinite(q).all():raise ValueError('Finite SDK31 target required')
        upper=q[OWNED]
        if np.any(upper<self.lower-1e-6) or np.any(upper>self.upper+1e-6):raise ValueError('Upper target outside hard bounds')
        if np.any(self.a@upper>self.b+1e-6):raise ValueError('Upper target outside coupled bounds')


def gains():
    try:import tomllib
    except ImportError:import tomli as tomllib
    profile=tomllib.loads(GAINS.read_text())['motion']['gains']
    kp=np.zeros(31);kd=np.zeros(31)
    for i in [0,1,2,*range(15,29)]:
        name=JOINTS[i].replace('l_','left_',1)if JOINTS[i].startswith('l_')else JOINTS[i].replace('r_','right_',1)if JOINTS[i].startswith('r_')else JOINTS[i]
        kp[i]=profile[name]['kp'];kd[i]=profile[name]['kd']
    kp[29:]=30.;kd[29:]=3.
    return kp,kd


def prepare(source,out,*,native_policy_clock=False,simulation_rate_limit=None):
    if simulation_rate_limit is not None and (not math.isfinite(simulation_rate_limit)or not 0<simulation_rate_limit<=1):
        raise ValueError('Finite simulation wall-rate cap in (0,1] required')
    out.mkdir(parents=True,exist_ok=False);trial=out/'run';trial.mkdir()
    for name in INPUT_FILES:shutil.copy2(source/name,trial/name)
    config=yaml.safe_load((trial/'robot.yaml').read_text())
    config['igris_c']['description']['mujoco']['robot_model_path']=str(trial/'scene.xml')
    (trial/'robot.yaml').write_text(yaml.safe_dump(config))
    manifest=json.loads((trial/'manifest.json').read_text())
    if not manifest.get('stage_room_after_preparation'):raise ValueError('Existing room parking hook required')
    manifest.update(vendor_walk_probe=True,empty_hand_only=True,room_must_remain_parked=True,vendor_policy_native_clock=native_policy_clock)
    manifest.pop('simulation_rate_limit',None)
    if simulation_rate_limit is not None:manifest['simulation_rate_limit']=simulation_rate_limit
    (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return trial


def carry_target(source,waist):
    path=next(source.glob('command_*.json'))
    sample=next(s for s in json.loads(path.read_text())['samples']if 'walking'in s)
    q=np.array(sample['motor_target'],float);q[0]=waist;q[29:]=0.
    return q,dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_wall=sample['wall'],
                  arm_coordinates='Actual published motor_target[15:29] at walking entry, held directly in SDK encoder coordinates',
                  nominal_target=sample['target'],original_motor_target=sample['motor_target'],target=q.tolist())


def worker(out,args):
    if not Path('/.dockerenv').exists():raise RuntimeError('Private simulator container required')
    runtime=json.loads((out/'runtime.json').read_text())
    if runtime['network']!='none' or runtime['ipc']!='private' or runtime['hardware_executed']:raise RuntimeError('Private SIM required')
    manifest=json.loads((out/'manifest.json').read_text())
    if (not manifest.get('vendor_walk_probe')or manifest.get('vendor_policy_native_clock')!=args.native_policy_clock
            or runtime.get('vendor_policy_native_clock')!=args.native_policy_clock):raise RuntimeError('Declared vendor policy clock probe required')
    os.sched_setaffinity(0,{args.worker_cpu})
    from run import latest
    from control_gc import defer_cyclic_gc
    sys.path.insert(0,str(ROOT))
    from igris.hl_ik.native_clock import read_clock
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd,LowState,RobotState,WalkTwist
    gate=UpperLimits();kp,kd=gains();goal,evidence=carry_target(args.source,args.waist)
    gate.check(goal)
    (out/'carry_target.json').write_text(json.dumps(evidence,indent=2)+'\n')
    rclpy.init();node=rclpy.create_node('vendor_empty_hand_waist_probe')
    state=None;mode=environment=None;received=mode_at=0.
    def on_state(message):
        nonlocal state,received
        state=dict(q=[float(j.q)for j in message.joint_state],dq=[float(j.dq)for j in message.joint_state],
                   error_code=[int(j.status_bits)for j in message.joint_state])
        received=time.monotonic()
    def on_mode(message):
        nonlocal mode,environment,mode_at
        mode=int(message.state);environment=int(message.environment);mode_at=time.monotonic()
    node.create_subscription(LowState,'igris_c_001/rt/lowstate',on_state,qos_profile_sensor_data)
    node.create_subscription(RobotState,'igris_c_001/rt/robotstate',on_mode,qos_profile_sensor_data)
    upper=node.create_publisher(LowCmd,'igris_c_001/rt/lowcmd',qos_profile_sensor_data)
    velocity=node.create_publisher(WalkTwist,'igris_c_001/rt/cmd_vel',10)
    def refresh():
        for _ in range(8):rclpy.spin_once(node,timeout_sec=0)
    def publish(q,command,seq):
        msg=LowCmd();msg.header.seq=seq;msg.kinematic_modes=[1]*5
        for i,m in enumerate(msg.motors):
            m.id=i;m.q=float(q[i])if i in OWNED else 0.;m.dq=0.;m.tau=0.;m.kp=float(kp[i]);m.kd=float(kd[i])
        upper.publish(msg)
        twistmsg=WalkTwist();twistmsg.header.seq=seq
        now=time.time();twistmsg.header.sec=int(now);twistmsg.header.nanosec=int((now%1)*1e9)
        twistmsg.twist.angular.z=float(command[2]);velocity.publish(twistmsg)
    samples=[];result=dict(pass_=False,error=None,hardware_executed=False,loaded=False,waist_sdk_rad=args.waist,
        policy_clock='native50Hz'if args.native_policy_clock else 'unchanged50Hzwall',
        simulation_rate_limit=manifest.get('simulation_rate_limit'),sequence_native_s=[3.,8.,2.],ramp_acceleration_rad_s2=.2)
    count=0;last_q=None
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            refresh()
            if state is not None and mode==6 and velocity.get_subscription_count():break
            time.sleep(.01)
        require_state(environment,mode,state,time.monotonic()-received,time.monotonic()-mode_at)
        if not velocity.get_subscription_count():raise RuntimeError('No reliable WalkTwist subscriber')
        (out/'upper_hold.stop').write_text('vendor probe acquiring upper LowCmd')
        deadline=time.monotonic()+2
        while not(out/'upper_hold.stopped').exists():
            refresh()
            if time.monotonic()>deadline:raise RuntimeError('Upper hold ownership timeout')
            time.sleep(.005)
        hold=json.loads((out/'upper_hold_result.json').read_text())
        if hold['error']:raise RuntimeError('Upper hold error: '+hold['error'])
        initial=np.asarray(state['q']).copy();initial[OWNED]=np.asarray(hold['q'])[OWNED]
        points,times=approach(initial,goal)
        for q in points:gate.check(q)
        result['approach_duration_s']=float(times[-1]);result['approach_points']=points.tolist()
        start=read_clock(out/'native_clock.bin')['sim'];wall_start=time.monotonic()
        metadata=json.loads((out/'model.json').read_text());torso=next(i for i,b in enumerate(metadata['bodies'])if b['name']=='torso')
        stamp,poses,pi=latest(out);origin=poses[pi].copy();probe_origin=None
        with defer_cyclic_gc():
            while time.monotonic()-wall_start<args.wall_timeout:
                refresh();require_state(environment,mode,state,time.monotonic()-received,time.monotonic()-mode_at)
                if (out/'stop.request').exists():raise RuntimeError((out/'stop.request').read_text())
                if (out/'room_layout.request').exists()or(out/'room_layout_applied.json').exists():raise RuntimeError('Unexpected room staging')
                reference=(out/'hand_reference.txt').read_text().split()
                if not 0<=time.time()-float(reference[0])<.25:raise RuntimeError('Stale MCP reference')
                stamp,poses,pi=latest(out);require_pose(poses[pi],origin)
                progress=read_clock(out/'native_clock.bin')['sim']-start
                elapsed=progress-times[-1]
                if elapsed>=0 and probe_origin is None:
                    if manifest.get('simulation_rate_limit') is not None:(out/'pacing.request').touch(exist_ok=False)
                    probe_origin=poses[pi].copy();result['probe_start_sim']=start+times[-1];result['probe_start_wall']=time.time()
                    print('PROBE_STARTED',args.waist,result['probe_start_sim'],flush=True)
                if probe_origin is not None:require_pose(poses[pi],probe_origin)
                target=interpolate(points,times,progress);gate.check(target)
                command=twist(elapsed)
                if elapsed>=13.:result['pass_']=True;break
                publish(target,command,count);last_q=target.copy()
                samples.append(dict(wall=time.time(),sim=float(stamp[1]),progress_s=progress,probe_elapsed_s=elapsed,
                    phase='approach'if elapsed<0 else 'hold'if elapsed<3 else 'turn'if elapsed<11 else 'stop',
                    q=state['q'],target=target.tolist(),command=command,pelvis=poses[pi].tolist(),torso=poses[torso].tolist(),
                    lowstate_age_s=time.monotonic()-received,robotstate_age_s=time.monotonic()-mode_at,mode=mode,environment=environment))
                count+=1;time.sleep(max(0.,wall_start+count*.01-time.monotonic()))
            else:raise RuntimeError('Probe wall timeout')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}';result['traceback']=traceback.format_exc()
    finally:
        # Zero twist and last acknowledged upper target; no cleanup motion.
        if last_q is not None:
            for i in range(10):publish(last_q,[0.,0.,0.],count+i);time.sleep(.01)
        else:
            for _ in range(3):velocity.publish(WalkTwist())
        node.destroy_node();rclpy.shutdown()
        result['pass']=result.pop('pass_');result['sample_count']=len(samples)
        (out/'probe_samples.json').write_text(json.dumps(samples,allow_nan=False))
        active=[s for s in samples if s['probe_elapsed_s']>=0]
        if active:
            a,b=active[0],active[-1];native=b['sim']-a['sim'];wall=b['wall']-a['wall']
            result.update(duration_native_s=native,duration_wall_s=wall,rtf=native/wall,
                pelvis_delta_xyz_m=(np.array(b['pelvis'][:3])-a['pelvis'][:3]).tolist(),
                pelvis_yaw_change_rad=float(np.unwrap([yaw(s['pelvis'])for s in active])[-1]-yaw(a['pelvis'])),
                pelvis_tilt_max_rad=max(math.acos(np.clip(1-2*(s['pelvis'][4]**2+s['pelvis'][5]**2),-1,1))for s in active))
        (out/'probe_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('PROBE_RESULT',json.dumps(result),flush=True)
    return 0 if result['pass']else 1


def launch(out,args):
    import run
    trial=prepare(args.source,out,native_policy_clock=args.native_policy_clock,simulation_rate_limit=args.simulation_rate_limit)
    environment=os.environ.copy()
    environment['IGRIS_HAND_PYTHON']=str(ROOT/'third_party/curobo/.venv/bin/python')
    def interrupted(*_):raise RuntimeError('Probe interrupted; stopping owned runtime')
    signal.signal(signal.SIGTERM,interrupted)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),HERE/'run.py',HERE/'upper_hold.py',HERE/'room_setup.cpp',HERE/'observer.cpp',
                 HERE/'vendor_policy_clock.cpp',HERE/'vendor_policy_clock.py',HERE/'simulation_pacer.cpp',GAINS]:shutil.copy2(path,snapshot/path.name)
    report=dict(pass_=False,error=None,hardware_executed=False,loaded=False,waist_sdk_rad=args.waist,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in snapshot.iterdir()},
        original_scene_sha256=hashlib.sha256((args.source/'scene.xml').read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256((trial/'scene.xml').read_bytes()).hexdigest())
    try:
        # Existing lifecycle owns model audit, private container and hand process.
        with(out/'start.log').open('w')as log:
            subprocess.run([sys.executable,str(HERE/'run.py'),'start','--out',str(trial),'--image',args.image],stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
        with(out/'bringup.log').open('w')as log:
            subprocess.run([sys.executable,str(HERE/'run.py'),'bringup','--out',str(trial),'--physics-cpu',str(args.physics_cpu)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150,env=environment)
        argv=['python3',str(Path(__file__).resolve()),'worker','--out',str(trial),'--source',str(args.source),'--waist',str(args.waist),
              '--worker-cpu',str(args.worker_cpu),'--wall-timeout',str(args.wall_timeout)]
        if args.native_policy_clock:argv+=['--native-policy-clock']
        with(out/'worker.log').open('w')as log:
            run.shell(trial,'exec '+shlex.join(argv),stdout=log,stderr=subprocess.STDOUT,timeout=args.wall_timeout+30)
        report['worker_result']=json.loads((trial/'probe_result.json').read_text());report['pass_']=report['worker_result']['pass']
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['traceback']=traceback.format_exc()
        if(trial/'probe_result.json').exists():report['worker_result']=json.loads((trial/'probe_result.json').read_text())
    finally:
        if(trial/'runtime.json').exists():
            try:
                # stop validates container label/network/IPC and exact MCP argv.
                stopped=subprocess.run([sys.executable,str(HERE/'run.py'),'stop','--out',str(trial)],capture_output=True,text=True,timeout=30)
                report['cleanup']=dict(returncode=stopped.returncode,stdout=stopped.stdout,stderr=stopped.stderr)
                if stopped.returncode:report['pass_']=False
            except Exception as exc:report['cleanup_error']=str(exc);report['pass_']=False
        report['pass']=report.pop('pass_');(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('VENDOR_WALK_PROBE',json.dumps(report),flush=True)
    return 0 if report['pass']else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','worker'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--waist',type=float,required=True);p.add_argument('--physics-cpu',type=int,default=1)
    p.add_argument('--worker-cpu',type=int,default=8);p.add_argument('--wall-timeout',type=float,default=240.)
    p.add_argument('--native-policy-clock',action='store_true')
    p.add_argument('--simulation-rate-limit',type=float)
    p.add_argument('--image',default='igris-sim:jazzy');args=p.parse_args()
    if not math.isfinite(args.waist)or abs(args.waist)>1.0:raise ValueError('Bounded waist yaw in[-1,1]required')
    if not math.isfinite(args.wall_timeout)or not 30<=args.wall_timeout<=600:raise ValueError('Bounded wall timeout required')
    args.source=args.source.resolve();args.out=args.out.resolve()
    return worker(args.out,args)if args.action=='worker'else launch(args.out,args)


if __name__=='__main__':raise SystemExit(main())
