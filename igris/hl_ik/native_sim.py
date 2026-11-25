"""Run learned-IK joint trajectories through an isolated native vendor simulator.

Network mode is `none`, IPC is private, and all DDS commands are issued inside
this experiment's container. Existing simulators and physical devices are unused.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
IGRIS=ROOT/'igris'
OUT=ROOT/'outputs/hl_ik_20260915'
SOURCE=IGRIS/'sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg'
CONTAINER='igris-hlik-sim-20260915'
OWNED=np.r_[0:3,15:29]

def run(args,**kwargs):return subprocess.run(args,check=True,**kwargs)
def shell(script,**kwargs):return run(['docker','exec',CONTAINER,'bash','-lc',script],**kwargs)

def initialize():
    if subprocess.run(['docker','inspect',CONTAINER],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        info=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]
        if info['HostConfig']['NetworkMode']!='none' or info['HostConfig']['IpcMode']!='private':raise RuntimeError('unexpected existing container isolation')
        return
    (OUT/'native').mkdir(exist_ok=True)
    (OUT/'native/cyclonedds.xml').write_text('''<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>100</MaxAutoParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>''')
    run(['docker','run','-d','--name',CONTAINER,'--network','none','--ipc','private','--shm-size','2g',
        '--ulimit','memlock=-1','--ulimit','rtprio=99','--cap-add','SYS_NICE','-e','DISPLAY=:99',
        '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{IGRIS}:/igris:ro','-v',f'{IGRIS}/configs:/robros/configs:ro',
        '-v',f'{OUT}:/hlik','-v',f'{OUT}/native/cyclonedds.xml:/robros/configs/cyclonedds/cyclonedds_local.xml:ro',
        'igris-sim:jazzy','sleep','infinity'],stdout=subprocess.DEVNULL)
    run(['docker','exec','-d',CONTAINER,'Xvfb',':99','-screen','0','1280x720x24'])
    for source,name in [('record_physics.cpp','observer.so'),('hand_motor.cpp','hand_motor.so')]:
        shell('g++ -O2 -std=c++17 -shared -fPIC -I/igris/ws/install/igris_c_mujoco/include '+
            shlex.quote('/igris/sim_local/vr_box_eval/'+source)+' -ldl -o '+shlex.quote('/hlik/native/'+name))
    info=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]
    (OUT/'native/isolation.json').write_text(json.dumps(dict(container=CONTAINER,image=info['Image'],network=info['HostConfig']['NetworkMode'],ipc=info['HostConfig']['IpcMode'],hardware_executed=False),indent=2))

def prepare(name,motion):
    import yaml
    import tomllib
    sys.path.insert(0,str(IGRIS/'igris_curobo/src'))
    from igris_ik.realtime.robot_control import VENDOR_HOLD_KP,VENDOR_HOLD_KD
    out=OUT/'native'/name;out.mkdir(exist_ok=False)
    for filename in ['scene.xml','compliance.json','vendor_params.yaml','replay_gains.toml','input.json']:
        shutil.copy2(SOURCE/filename,out/filename)
    shutil.copy2(__file__,out/'native_sim.py')
    shutil.copytree(SOURCE/'motions',out/'motions')
    cfg=yaml.safe_load((SOURCE/'robot.yaml').read_text());c=cfg['igris_c']
    c['description']['mujoco']['robot_model_path']=f'/hlik/native/{name}/scene.xml'
    c['controller']['motion_root_dir']=f'/hlik/native/{name}/motions'
    c['controller']['gain_parameter_path']=f'/hlik/native/{name}/vendor_params.yaml'
    (out/'robot.yaml').write_text(yaml.safe_dump(cfg))
    shutil.copy2(motion,out/'motion.npz')
    with np.load(motion) as f:
        assert f['q'].shape[1]==31 and np.isfinite(f['q']).all()
        assert np.all(np.diff(f['time'])>0)
        scale=float(f['time_scale']) if 'time_scale' in f else 1.
    if scale!=1.:
        inputs=json.loads((out/'input.json').read_text())
        for frame in inputs:
            frame['original_task_time_s']=frame['t'];frame['t']*=scale
        (out/'input.json').write_text(json.dumps(inputs))
    gains=tomllib.loads((SOURCE/'replay_gains.toml').read_text())['motion']['gains']
    names=['waist_yaw','waist_roll','waist_pitch']+[f'{s}_{j}' for s in ('left','right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch')]
    meta=json.loads((SOURCE/'metadata.json').read_text())
    initial=json.loads((SOURCE/'motions/HOME.json').read_text())['motions'][0]['position']
    ready_kd=VENDOR_HOLD_KD.copy();ready_kd[15:29]=np.minimum(ready_kd[15:29],5.)
    (out/'trial.json').write_text(json.dumps(dict(name=name,gains=[[gains[n]['kp'],gains[n]['kd']] for n in names],
        ready_gains=np.column_stack((VENDOR_HOLD_KP[OWNED],ready_kd[OWNED])).tolist(),initial_q=initial,
        source=str(SOURCE),source_metadata=meta,hardware_executed=False,time_scale=scale,motion_clock='native simulation time',model='original native free-base 1 kg box scene; learned torso-arm IK commands'),indent=2))
    return out

def latest(path):
    with path.open('rb') as f:
        header=f.read(8)
        if len(header)<8:return None
        nq,nv=struct.unpack('ii',header);width=2+nq+nv+7;n=(f.seek(0,2)-8)//(width*8)
        if n<1:return None
        f.seek(8+(n-1)*width*8);return np.frombuffer(f.read(width*8),np.float64)

def boost_physics(out):
    if not Path('/.dockerenv').exists():raise RuntimeError('physics scheduling is container-local')
    processes=[p for p in Path('/proc').glob('[0-9]*') if (p/'comm').exists() and (p/'comm').read_text().strip()=='igris_c_mujoco']
    if len(processes)!=1:raise RuntimeError('expected exactly one owned native simulator')
    def ticks():
        return {int(p.name):sum(map(int,(p/'stat').read_text().split()[13:15])) for p in (processes[0]/'task').iterdir()}
    before=ticks();time.sleep(1);after=ticks();tid=max(after,key=lambda t:after[t]-before.get(t,0))
    if after[tid]-before.get(tid,0)<10:raise RuntimeError('native physics thread was not busy enough to identify')
    run(['chrt','-f','-p','50',str(tid)])
    (out/'physics_scheduling.json').write_text(json.dumps(dict(native_pid=int(processes[0].name),tid=tid,policy='SCHED_FIFO',priority=50,
        method='same busiest-thread rule as existing sim_local/rt_physics.sh; only this private container'),indent=2))

def trial(out):
    initialize();name=out.name
    active=shell("pgrep -af '^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch igris_c_controller simulation.launch.py' || true",capture_output=True,text=True).stdout
    if active:raise RuntimeError('owned simulator still active; terminate the previous owned launch before next trial')
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    shell('g++ -O2 -std=c++17 -shared -fPIC -I/igris/ws/install/igris_c_mujoco/include '
          '/workspace/igris/hl_ik/native_clock.cpp -ldl -o /hlik/native/native_clock.so')
    shutil.copy2(IGRIS/'hl_ik/native_clock.cpp',out/'native_clock.cpp')
    cp=f'/hlik/native/{name}'
    launch=f'''source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
cd /tmp
export DISPLAY=:99 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_PRELOAD=/hlik/native/native_clock.so:/hlik/native/hand_motor.so:/hlik/native/observer.so
export IGRIS_HLIK_CLOCK={shlex.quote(cp+'/native_clock.bin')}
export IGRIS_HAND_DIR={shlex.quote(cp)} IGRIS_EVAL_TRIGGER={shlex.quote(cp+'/record.trigger')} IGRIS_EVAL_LOG={shlex.quote(cp+'/physics.bin')}
exec ros2 launch igris_c_controller simulation.launch.py config_file:={shlex.quote(cp+'/robot.yaml')} > {shlex.quote(cp+'/vendor.log')} 2>&1'''
    run(['docker','exec','-d',CONTAINER,'bash','-lc',launch])
    time.sleep(8)
    def ctl(args):
        cmd='source /opt/ros/jazzy/setup.bash\nsource /igris/ws/install/setup.bash\nsource /igris/sim_local/tools_ws/install/setup.bash\nexec ros2 run igris_sim_ctl igris_ctl '+shlex.join(args)+' --cyclonedds-xml /robros/configs/cyclonedds/cyclonedds_local.xml'
        return subprocess.Popen(['docker','exec',CONTAINER,'bash','-lc',cmd],stdout=bringup,stderr=bringup)
    with (out/'bringup.log').open('w') as bringup,(out/'worker.log').open('w') as log:
        env='source /opt/ros/jazzy/setup.bash\nsource /igris/ws/install/setup.bash\nexport PYTHONPATH=/workspace:${PYTHONPATH:-} CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml\nexec python3 '+shlex.quote(cp+'/native_sim.py')+' --worker '+shlex.quote(cp)
        worker_process=subprocess.Popen(['docker','exec',CONTAINER,'bash','-lc',env],stdout=log,stderr=log)
        hand_env={**os.environ,'PYTHONPATH':str(IGRIS/'igris_curobo/src'),'OPENBLAS_NUM_THREADS':'1'}
        hand_log=(out/'hand_controller.log').open('w')
        hand_process=None
        try:
            for args in [('sim','resume'),('torque','on'),('mode','home')]:
                if ctl(args).wait()!=0:raise RuntimeError(f'simulation service failed {args}')
            shell('PYTHONPATH=/workspace python3 -m igris.hl_ik.native_sim --boost '+shlex.quote(cp),stdout=bringup,stderr=bringup)
            time.sleep(2)
            reset=ctl(('sim','reset'));time.sleep(.08);walk=ctl(('mode','lowwalk'))
            if reset.wait() or walk.wait():raise RuntimeError('native simulation reset/lowwalk failed')
            time.sleep(5) # Same passive-finger reset settling interval as the source experiment.
            # Start the freshness-monitored hand loop only once reset has ended.
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                motor=out/'hand_motor.bin'
                if motor.exists() and motor.stat().st_size>=23*8:
                    with motor.open('rb') as f:
                        f.seek(-23*8,2);stamp=np.frombuffer(f.read(23*8),np.float64)[0]
                    if 0<=time.time()-stamp<.05:break
                time.sleep(.01)
            else:raise RuntimeError('no fresh native hand state after reset')
            hand_process=subprocess.Popen(['/home/rcv/workspace/curobo/.venv/bin/python',str(IGRIS/'sim_local/vr_box_eval/compliant_hands.py'),'--run',str(out)],env=hand_env,stdout=hand_log,stderr=hand_log)
            code=worker_process.wait(timeout=240)
            if code:raise RuntimeError(f'native worker exit {code}')
        finally:
            ctl(('sim','pause')).wait()
            if hand_process is not None:hand_process.terminate();hand_process.wait(timeout=10)
            hand_log.close()
            if worker_process.poll() is None:worker_process.terminate()
            shell("pkill -INT -f '^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch igris_c_controller simulation.launch.py' || true")

def worker(out):
    if not Path('/.dockerenv').exists() or not str(out).startswith('/hlik/native/'):
        raise RuntimeError('worker is restricted to this isolated simulator')
    sys.path.insert(0,'/igris/igris_curobo/tools')
    from dds_zmq_bridge import _ros_cmd_writer,_state_types,lowstate_to_dict
    from cyclonedds.core import Policy,Qos
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader,Subscriber
    from cyclonedds.topic import Topic
    from igris.hl_ik.native_clock import read_clock
    cfg=json.loads((out/'trial.json').read_text());g=np.asarray(cfg['gains']);ready_g=np.asarray(cfg['ready_gains']);initial=np.asarray(cfg['initial_q'])
    z=np.load(out/'motion.npz');times=z['time'];commands=z['q']
    depth=1 if cfg.get('online_core') else 20
    qos=Qos(Policy.Reliability.BestEffort,Policy.Durability.Volatile,Policy.History.KeepLast(depth))
    participant=DomainParticipant(0);sub=Subscriber(participant);st,mt=_state_types('/igris')
    sr=DataReader(sub,Topic(participant,'igris_c_001/rt/lowstate',st,qos=qos),qos=qos)
    mr=DataReader(sub,Topic(participant,'igris_c_001/rt/robotstate',mt,qos=qos),qos=qos)
    send=_ros_cmd_writer('igris_c_001','/igris')
    cmd=dict(q=[0.]*31,dq=[0.]*31,kp=[0.]*31,kd=[0.]*31,kinematic_modes=[1]*5)
    def set_gains(values):
        for i,(kp,kd) in zip(OWNED,values):cmd['kp'][i]=float(kp);cmd['kd'][i]=float(kd)
    set_gains(ready_g)
    state=None;mode=None;last_state=0;records=[];seq=0
    def refresh():
        nonlocal state,mode,last_state
        for m in mr.take(N=1 if cfg.get('online_core') else 30):mode=m.state.name
        for s in sr.take(N=1 if cfg.get('online_core') else 50):state=lowstate_to_dict(s,'igris_c_001',time.time(),mode);last_state=time.monotonic()
        return state,mode,time.monotonic()-last_state
    def publish(q,phase,at):
        nonlocal seq
        refresh()
        for i in OWNED:cmd['q'][i]=float(q[i])
        send(cmd,seq);seq+=1
        records.append([time.time(),at,phase,*q])
    result=dict(completed=False,hardware_executed=False,error=None)
    (out/'record.trigger').write_text(str(time.time()))
    try:
        start=time.monotonic();stable=None
        while time.monotonic()-start<45:
            publish(initial,0,-1);row=latest(out/'physics.bin') if (out/'physics.bin').exists() else None
            if row is not None and mode=='ROBOT_STATE_WALK_LOW' and state is not None:
                bp=row[2:9];tilt=np.arccos(np.clip(1-2*(bp[4]**2+bp[5]**2),-1,1))
                good=bp[2]>.7 and tilt<np.deg2rad(10) and np.max(np.abs(np.asarray(state['q'])[OWNED]-initial[OWNED]))<.15
                stable=time.monotonic() if good and stable is None else stable if good else None
                if stable is not None and time.monotonic()-stable>2:break
            time.sleep(.01)
        else:raise RuntimeError(f'initial stabilization failed; mode={mode}')
        # Finite approach in joint space; no state assignment after scene reset.
        qstart=np.asarray(state['q']);approach=time.monotonic()
        while (age:=time.monotonic()-approach)<10:
            u=min(age/4,1);s=u**3*(10-15*u+6*u*u)
            gain_u=np.clip((age-5)/2,0,1);set_gains(ready_g+gain_u*(g-ready_g))
            publish(qstart+s*(commands[0]-qstart),1,-1);time.sleep(.01)
        if cfg.get('online_core'):
            from igris.hl_ik.core_online import run_native
            run_native(out,refresh,publish,commands[0],result)
            result['completed']=True
            return
        start_wall=time.time();start=time.monotonic();result['start_wall']=start_wall;index=0;tick=0
        start_sim=read_clock(out/'native_clock.bin')['sim'];result['start_sim']=start_sim;result['motion_clock']='every-step native shared clock'
        previous_age=-1.
        while (age:=read_clock(out/'native_clock.bin')['sim']-start_sim)<=float(times[-1]):
            if age<previous_age:raise RuntimeError('native clock moved backwards')
            previous_age=age
            while index+1<len(times) and times[index+1]<age:index+=1
            following=min(index+1,len(times)-1)
            alpha=0. if following==index else np.clip((age-times[index])/(times[following]-times[index]),0,1)
            q=(1-alpha)*commands[index]+alpha*commands[following];publish(q,2,age)
            if time.monotonic()-last_state>.25:raise RuntimeError('stale native lowstate')
            if (out/'stop.request').exists():raise RuntimeError((out/'stop.request').read_text())
            row=latest(out/'physics.bin')
            if row is not None:
                if time.time()-row[0]>.15:raise RuntimeError('stale native physics')
                if row[4]<.65:raise RuntimeError('native base below 0.65 m')
                if row[-4]>0:raise RuntimeError('native self-contact detected')
            tick+=1;deadline=start+tick*.01
            if time.monotonic()-deadline>.05:raise RuntimeError('native command publisher more than 50 ms late')
            time.sleep(max(0,deadline-time.monotonic()))
        while (age:=read_clock(out/'native_clock.bin')['sim']-start_sim)<=float(times[-1])+2:
            publish(commands[-1],3,age)
            if (out/'stop.request').exists():raise RuntimeError((out/'stop.request').read_text())
            time.sleep(.01)
        result['completed']=True
    except Exception as e:result['error']=str(e)
    finally:
        np.savez_compressed(out/'commands.npz',records=np.asarray(records));(out/'result.json').write_text(json.dumps(result,indent=2))
        if 'start_wall' in result:
            meta=cfg['source_metadata'];meta.update(start_wall=result['start_wall'],duration=time.time()-result['start_wall'],
                input=('online HL-IK 30 Hz; igris_core C2 follow_stream at 100 Hz; native simulation progress' if cfg.get('online_core') else 'HL-IK 30 Hz generated history; interpolated commands with 100 Hz wall-clock publisher; progress follows native simulation time'),hardware_executed=False)
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            moving=np.asarray([r for r in records if r[2]>=2]);progress,indices=np.unique(moving[:,1],return_index=True)
            wall=moving[indices,0]-result['start_wall'];source_frames=json.loads((out/'input.json').read_text());mapped=[]
            for frame in source_frames:
                if frame['t']>progress[-1]:break
                frame['source_motion_time_s']=frame['t'];frame['t']=float(np.interp(frame['t'],progress,wall));mapped.append(frame)
            (out/'input.json').write_text(json.dumps(mapped))
    print(result,flush=True)
    if result['error']:raise SystemExit(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');ap.add_argument('--prepare');ap.add_argument('--motion',type=Path)
    ap.add_argument('--trial',type=Path);ap.add_argument('--worker',type=Path);ap.add_argument('--boost',type=Path);a=ap.parse_args()
    if a.initialize:initialize()
    elif a.prepare:print(prepare(a.prepare,a.motion))
    elif a.trial:trial(a.trial)
    elif a.worker:worker(a.worker)
    elif a.boost:boost_physics(a.boost)

if __name__=='__main__':main()
