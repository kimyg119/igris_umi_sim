"""Online HL-IK / igris_core experiment, restricted to the private native sim.

IK, exact configured sphere checking and the 100 Hz core executor run in
separate processes. Only EE goals and initial q are supplied before execution.
The prior history contains the solver's own 30 Hz solutions (not measured q).
Measured full posture is used for each outgoing command's collision corridor.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
import select
import shutil
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hl_ik_20260915'
OWNED=np.r_[0:3,15:29]


class Channel:
    """One outstanding request per Unix datagram channel; no target backlog."""
    def __init__(self,name,server=False):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        local=name+('.sock' if server else '_client.sock')
        self.sock.bind(local)
        if os.geteuid()==0:
            owner=Path('.').stat();os.chown(local,owner.st_uid,owner.st_gid)
        os.chmod(local,0o600)
        self.peer=name+('_client.sock' if server else '.sock')
    def send_json(self,value):self.sock.sendto(json.dumps(value,allow_nan=False).encode(),self.peer)
    def recv_json(self):return json.loads(self.sock.recv(32768))
    def poll(self,milliseconds):return bool(select.select([self.sock],[],[],milliseconds/1000)[0])


def service(out, kind):
    from .solver import HLIK
    from .pipeline_experiment import make_solver
    os.chdir(out)
    fixture=np.load('targets.npz');reference=fixture['reference'];goals=fixture['goals']
    solver=make_solver('qp_priority_full',box=True) if kind=='ik' else HLIK(hand_envelope=out/'compliance.json')
    q=reference[0].copy();solver.reset(q)
    if kind=='ik':
        for _ in range(3):solver.solve(goals[0],q)
        solver.reset(q)
    sock=Channel(kind,server=True)
    (out/(kind+'.ready')).write_text(str(os.getpid()))
    cfg=json.loads((out/'trial.json').read_text())
    import gc
    gc_events=[]
    gc.callbacks.append(lambda phase,info:gc_events.append([time.monotonic(),phase,info['generation']]))
    if cfg.get('defer_gc'):
        gc.collect();gc.disable()  # This child exits after the bounded trial.
    with (out/(kind+'.jsonl')).open('w',buffering=1) as log:
        while True:
            req=sock.recv_json();begin=time.monotonic();gc_begin=len(gc_events)
            try:
                if kind=='ik':
                    i=int(req['frame'])
                    if i<0 or i>=len(goals):raise ValueError('goal frame outside fixture')
                    previous=q.copy();q[:15]=reference[i,:15]
                    if i==cfg.get('inject_frame',-1):time.sleep(cfg['inject_delay_s'])
                    r=solver.solve(goals[i],q,record_history=False)
                    if r.accepted:q=r.q;solver.observe(q)
                    else:q=previous
                    result=dict(frame=i,q=q.tolist(),accepted=r.accepted,reason=r.reason,
                        solve_ms=r.elapsed_ms,position_m=r.position_error_m,orientation_rad=r.orientation_error_rad,
                        qp_ms=r.qp_ms,collision_ms=r.collision_ms,qp_fallbacks=r.qp_fallbacks,
                        requested_wall=req['wall'],goal_time=float(fixture['time'][i]))
                else:
                    a,b=np.asarray(req['previous']),np.asarray(req['candidate'])
                    if a.shape!=(31,) or b.shape!=(31,) or not np.isfinite([a,b]).all():
                        raise ValueError('finite full SDK posture required')
                    a[29:]=0.;b[29:]=0.
                    arm=b[15:29]
                    if np.any(arm<solver.limits[:,0]-1e-8) or np.any(arm>solver.limits[:,1]+1e-8) or np.any(solver.A@arm>solver.b+1e-8):
                        raise ValueError('interpolated command violates arm limits')
                    solver.set_q(a);qa=solver.configuration.q.copy()
                    solver.set_q(b);qb=solver.configuration.q.copy()
                    hit=solver.geometry.corridor(solver.configuration.data,qa,qb,max_step=.005)
                    result=dict(accepted=not bool(hit),reason=hit)
                result.update(begin=begin,end=time.monotonic())
            except Exception as e:result=dict(accepted=False,reason=str(e),begin=begin,end=time.monotonic())
            result['gc_events']=gc_events[gc_begin:]
            sock.send_json(result);log.write(json.dumps(result)+'\n')


def run_native(out, refresh, publish, initial, result):
    """Called only after the existing native stabilization and hand bringup."""
    if not Path('/.dockerenv').exists() or not str(out).startswith('/hlik/native/'):
        raise RuntimeError('online executor is private-simulator only')
    sys.path.insert(0,'/workspace/igris-core/igris_core')
    from igris_core import Motion,URDF_JOINT_NAMES
    from igris_core.joint_trajectory import OnlineJointTrajectory
    from .native_clock import read_clock
    from .native_sim import latest
    os.chdir(out)
    cfg=json.loads((out/'trial.json').read_text());fixture=np.load('targets.npz')
    import atexit
    monitor_log=(out/'dds_monitor.log').open('w')
    monitor_process=subprocess.Popen([sys.executable,'-B','-m','igris.hl_ik.core_online','--monitor','--out',str(out)],stdout=monitor_log,stderr=monitor_log)
    def stop_monitor():
        monitor_process.terminate();monitor_process.wait(timeout=5);monitor_log.close()
    atexit.register(stop_monitor)
    ik=Channel('ik');guard=Channel('guard')
    gains={'motion':{'gains':{URDF_JOINT_NAMES[i]:{'kp':g[0],'kd':g[1]} for i,g in zip(OWNED,cfg['gains'])}}}
    phase=1;progress=-1.;start_sim=None;trace=[];updates=[]

    class Backend:
        def clock(self):return time.monotonic()
        def sleep_until(self,t):time.sleep(max(0,t-time.monotonic()))
        def acquire(self):pass  # The single native worker already owns this isolated publisher.
        def release(self):pass
        def snapshot(self):
            state,mode,age=refresh()
            return dict(fresh=state is not None and age<.15,
                mode='WALK_LOW' if mode=='ROBOT_STATE_WALK_LOW' else mode,
                joint_q=state['q'],joint_dq=state['dq'])
        def publish(self,cmd):
            state,_,_=refresh();q=np.asarray(state['q']).copy()
            q[OWNED]=np.asarray(cmd['q'])[OWNED];q[29:]=0
            actual_progress=read_clock(out/'native_clock.bin')['sim']-start_sim if start_sim is not None else -1.
            publish(q,phase,actual_progress)

    class Guard:
        def command_guard(self,a,b):
            if (out/'stop.request').exists():return (out/'stop.request').read_text()
            row=latest(out/'physics.bin')
            if row is None or time.time()-row[0]>.15:return 'stale native physics'
            if row[4]<.65 or row[-4]>0:return 'native fall or self-contact'
            guard.send_json(dict(previous=a.tolist(),candidate=b.tolist()))
            if not guard.poll(20):raise RuntimeError('sphere guard response exceeded 20 ms')
            reply=guard.recv_json()
            return None if reply['accepted'] else str(reply['reason'])

    backend=Backend();curve=OnlineJointTrajectory(initial);pending=False;last_frame=-1;last_result=0.;previous_goal=None
    with Motion.walk_low(backend,gains=gains,safety=Guard(),max_velocity_rad_s=3.,tracking_error_limit_rad=.5) as motion:
        motion.servo({URDF_JOINT_NAMES[i]:float(initial[i]) for i in OWNED},duration=2.)
        import gc
        gc_events=[];gc_was_enabled=gc.isenabled()
        def gc_event(phase,info):gc_events.append([time.monotonic(),phase,info['generation']])
        gc.callbacks.append(gc_event)
        if cfg.get('defer_gc'):
            # Bounded simulator trial only. Reference counting stays enabled;
            # defer cyclic collection until the 100 Hz command stream ends.
            gc.collect();gc.disable()
        start_sim=read_clock(out/'native_clock.bin')['sim'];start_wall=time.time()
        result.update(start_sim=start_sim,start_wall=start_wall,motion_clock='native clock; core deadlines on monotonic wall clock')
        duration=float(fixture['time'][-1]);last_progress=-1.
        def sample(_wall_elapsed):
            nonlocal phase,progress,pending,last_frame,last_result,last_progress,previous_goal
            now=time.monotonic();progress=read_clock(out/'native_clock.bin')['sim']-start_sim
            if progress<last_progress:raise RuntimeError('native clock moved backwards')
            last_progress=progress;phase=2 if progress<=duration else 3
            if progress>duration+2:return None
            if ik.poll(0):
                reply=ik.recv_json();pending=False
                if not reply['accepted']:raise RuntimeError('online IK rejected: '+str(reply['reason']))
                target=np.asarray(reply['q']);target[3:15]=initial[3:15];target[29:]=initial[29:]
                velocity=np.zeros(31)
                if previous_goal is not None:
                    velocity=(target-previous_goal[1])/(reply['goal_time']-previous_goal[0])
                previous_goal=(reply['goal_time'],target.copy())
                # Anchor the endpoint to source time, not variable IK arrival.
                # Late results retain at least one source period to reconnect.
                segment_s=max(1/30,reply['goal_time']+cfg['horizon_s']-progress)
                curve.retarget(progress,target,segment_s,velocity=velocity)
                last_result=progress
                updates.append(dict(**reply,applied_progress=progress,applied_wall=time.time(),
                    target_velocity=velocity.tolist(),interpolation_duration_s=segment_s))
            frame=min(int(progress*30+1e-8),len(fixture['time'])-1)
            if frame>last_frame and not pending:
                ik.send_json(dict(frame=frame,wall=time.time()));pending=True;last_frame=frame
            if (progress<=duration or pending) and progress-last_result>.2:
                raise RuntimeError('online IK stale for 200 ms; stopping segment has finished')
            q=curve.sample(progress);v=curve.sample(progress,1);a=curve.sample(progress,2);j=curve.sample(progress,3)
            if np.max(abs(v[OWNED]))>3.+1e-8 or np.max(abs(a[OWNED]))>30.+1e-8:
                raise RuntimeError('online interpolation exceeds velocity/acceleration bound')
            trace.append([time.time(),progress,last_frame,time.monotonic()-now,*q,*v,*a,*j])
            return q
        try:
            motion.follow_stream(sample)
        finally:
            gc.callbacks.remove(gc_event)
            if gc_was_enabled:gc.enable()
            (out/'gc_events.json').write_text(json.dumps(gc_events))
            (out/'core_trace.json').write_text(json.dumps(dict(status=motion.status,trace=motion.trace)))
            (out/'ik_updates.json').write_text(json.dumps(updates))
            np.savez_compressed(out/'online.npz',records=np.asarray(trace))


def experiment(name,horizon=.1,inject_frame=-1,inject_delay=0.,defer_gc=False):
    from . import native_sim
    from .box_plan import box_targets
    from .common import sha256
    if not isinstance(name,str) or not re.fullmatch(r'[a-zA-Z0-9_-]+',name):raise ValueError('simple experiment name required')
    if not np.isfinite([horizon,inject_delay]).all() or horizon<=0 or inject_delay<0:
        raise ValueError('positive horizon and nonnegative injected delay required')
    # prepare() uses the existing 3.11 simulator environment. No arm solution
    # except q0 is reused: motion.npz is only a bootstrap pose/duration fixture.
    ts,goals,reference,_=box_targets()
    out=OUT/'native'/name
    if out.exists():raise FileExistsError(out)
    staging=ROOT/'outputs/hl_ik_core_online_20260915'/name;staging.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(staging/'bootstrap.npz',time=np.array([0.,ts[-1]]),q=np.stack([reference[0],reference[0]]))
    subprocess.run(['/home/rcv/workspace/curobo/.venv/bin/python','-m','igris.hl_ik.native_sim','--prepare',name,'--motion',str(staging/'bootstrap.npz')],check=True)
    np.savez_compressed(out/'targets.npz',time=ts,goals=goals,reference=reference)
    cfg=json.loads((out/'trial.json').read_text());cfg.update(online_core=True,horizon_s=horizon,inject_frame=inject_frame,inject_delay_s=inject_delay,defer_gc=defer_gc,
        interpolation='C3 seventh-degree segments; causal target velocity; source-time endpoint; C3 braking tail on missing updates',
        history='own accepted IK postures at nominal 30 Hz; measured posture used for outgoing full-posture collision checks',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'igris/hl_ik/native_sim.py',ROOT/'igris-core/igris_core/igris_core/joint_trajectory.py',ROOT/'igris-core/igris_core/igris_core/walk_motion.py',ROOT/'igris/hl_ik/pipeline.py']})
    (out/'trial.json').write_text(json.dumps(cfg,indent=2))
    for relative in cfg['source_sha256']:
        dst=out/'source'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dst)
    children=[];logs=[]
    try:
        for kind in ('ik','guard'):
            log=(out/(kind+'.log')).open('w');logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-B','-m','igris.hl_ik.core_online','--service',kind,'--out',str(out)],stdout=log,stderr=log))
        deadline=time.monotonic()+60
        while not all((out/(k+'.ready')).exists() for k in ('ik','guard')):
            if time.monotonic()>deadline or any(c.poll() is not None for c in children):raise RuntimeError('online services failed to initialize')
            time.sleep(.1)
        native_sim.trial(out)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait(timeout=10)
        for log in logs:log.close()
    print(out)


def monitor(out):
    """Receive the actual ROS LowCmd route over the same CycloneDDS transport.

    bridge_sim subscribes to this ROS type and consumes commands through SHM;
    it does not republish them as SDK-native LowCmd. The native-type topic has
    no writer in this experiment (confirmed using DDS/ROS discovery).
    """
    if not Path('/.dockerenv').exists():raise RuntimeError('container-only DDS monitor')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd
    rclpy.init();node=rclpy.create_node('hlik_lowcmd_observer')
    with (out/'dds_received.jsonl').open('w',buffering=1) as log:
        def receive(message):
            log.write(json.dumps([time.time(),time.monotonic(),message.header.seq,
                *[message.motors[int(i)].q for i in OWNED]])+'\n')
        node.create_subscription(LowCmd,'igris_c_001/rt/lowcmd',receive,qos_profile_sensor_data)
        rclpy.spin(node)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name');ap.add_argument('--horizon',type=float,default=.1)
    ap.add_argument('--service',choices=['ik','guard']);ap.add_argument('--out',type=Path)
    ap.add_argument('--inject-frame',type=int,default=-1);ap.add_argument('--inject-delay',type=float,default=0.)
    ap.add_argument('--monitor',action='store_true')
    ap.add_argument('--defer-gc',action='store_true')
    args=ap.parse_args()
    if args.monitor:monitor(args.out)
    elif args.service:service(args.out,args.service)
    else:experiment(args.name,args.horizon,args.inject_frame,args.inject_delay,args.defer_gc)
