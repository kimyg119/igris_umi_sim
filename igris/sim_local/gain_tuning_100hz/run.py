"""100 Hz replay gain experiment in the isolated vendor igris-sim container.

Host: --prepare CAMPAIGN creates immutable trial inputs. --trial DIR runs one.
Container: --worker DIR publishes only to the hard-coded simulation namespace.
No hardware endpoint argument is accepted. Original q and gains are never edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
IGRIS = HERE.parents[1]
OWNED = np.r_[0:3,15:29]
NAMES = ['waist_yaw','waist_roll','waist_pitch'] + [f'{s}_{j}' for s in ('l','r') for j in ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll')] + [f'{s}_{j}' for s in ('l','r') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch')] + ['neck_yaw','neck_pitch']

def container_path(path):
    return '/igris/' + str(Path(path).resolve().relative_to(IGRIS))

def shell(command, **kwargs):
    return subprocess.run(['docker','exec','igris-sim','bash','-lc',command],check=True,**kwargs)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(campaign,seed=None):
    import yaml
    import tomllib
    campaign.mkdir(parents=True,exist_ok=False)
    if seed is not None:
        shutil.copy2(seed,campaign/'seed_qpos.npy')
        (campaign/'seed.json').write_text(json.dumps(dict(source=str(seed.resolve()),sha256=sha(seed)),indent=2))
    source=IGRIS.parent/'outputs/synthetic_box_motion_20260914/motion_100hz.npz'
    shutil.copy2(source,campaign/'motion.npz')
    shutil.copy2(IGRIS/'real_client/replay_gains.toml',campaign/'original_replay_gains.toml')
    z=np.load(source,allow_pickle=False)
    assert np.array_equal(z['command_sdk_indices'],OWNED)
    assert np.allclose(np.diff(z['command_t_s']),.01,rtol=0,atol=1e-9)
    assert z['q_command_rad'].shape==(8061,17) and np.isfinite(z['q_command_rad']).all()
    assert np.all(z['mode']=='FOLLOWING') and np.all(z['trip']=='')
    cfg=tomllib.loads((campaign/'original_replay_gains.toml').read_text())['motion']['gains']
    long_names=[('left_'+n[2:] if n.startswith('l_') else 'right_'+n[2:] if n.startswith('r_') else n) for n in np.array(NAMES)[OWNED]]
    baseline=[[cfg[n]['kp'],cfg[n]['kd']] for n in long_names]
    profiles={'current':baseline}
    profiles['current_d5']=[[kp,5.] for kp,kd in baseline]
    # Deliberately bounded experimental grid, not manufacturer gain ceilings.
    for name,waist,arm,wrist,damp in [
        ('medium',250,[200,200,150,200],150,4.),
        ('medium_d5',250,[200,200,150,200],150,5.),
        ('stiff_d5',350,[350,300,250,350],300,5.),
        ('soft',200,[150,150,100,150],80,3.),
    ]:
        profiles[name]=[[waist,damp]]*3+([[x,damp] for x in arm]+[[wrist,damp]]*3)*2
    for name,waist,arm in [
        ('hybrid',350,[400,250,300,350,150,150,400]),
        ('precision',400,[500,300,400,500,200,200,400]),
        ('balanced_w400',350,[400,250,300,350,400,400,400]),
        ('precision_w400',400,[500,300,400,500,400,400,400]),
        ('upper_bound',500,[600,500,600,600,600,600,600]),
    ]:
        profiles[name]=[[waist,5.]]*3+[[x,5.] for x in arm]*2
    for kd in (2.,3.):
        gains=[pair.copy() for pair in profiles['precision_w400']]
        for j in (7,8,9,14,15,16):gains[j][1]=kd
        profiles[f'precision_wd{int(kd)}']=gains
    (campaign/'profiles.json').write_text(json.dumps(profiles,indent=2))
    original_files=[source,IGRIS/'real_client/replay_gains.toml',IGRIS/'sim_local/igris_c_v2_rcv_hand.xml',IGRIS/'sim_local/robot.yaml',IGRIS/'configs/controller/params.yaml',IGRIS/'igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf']
    manifest=dict(source=str(source),hashes={str(p):sha(p) for p in original_files},
        samples=len(z['command_t_s']),duration_s=float(z['command_t_s'][-1]),
        command_hz=100,dq_command=0,tau_command=0,owned_indices=OWNED.tolist(),
        payload='total bilateral load, equal split at each end_effector frame, rigid inertial load, no grasp claim',
        payload_gc='nominal vendor gravity compensation unchanged; payload is not added to controller URDF',
        kd_max=5,hardware_executed=False)
    (campaign/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for mass in (1.,2.):
        for name,gains in profiles.items():
            prepare_trial(campaign,name,mass,gains,z)
    print(campaign,flush=True)

def prepare_trial(campaign,name,mass,gains,z=None,offset=0.):
    import yaml
    if z is None:z=np.load(campaign/'motion.npz',allow_pickle=False)
    out=campaign/f'{name}_{mass:g}kg'
    out.mkdir(exist_ok=False)
    tree=ET.parse(IGRIS/'sim_local/igris_c_v2_rcv_hand.xml');root=tree.getroot()
    for side in ('left','right'):
        parent=root.find(f'.//body[@name="end_effector_{side}"]')
        payload=ET.SubElement(parent,'body',name=f'gain_payload_{side}',pos=f'0 0 {-offset}')
        inertia=.4*(mass/2)*.04**2
        ET.SubElement(payload,'inertial',pos='0 0 0',mass=str(mass/2),diaginertia=f'{inertia} {inertia} {inertia}')
        ET.SubElement(payload,'geom',type='sphere',size='.04',contype='0',conaffinity='0',group='1',rgba='.9 .65 .1 .5',mass='0')
    address=0;addresses={}
    for j in root.find('worldbody').iter():
        if j.tag not in ('joint','freejoint'):continue
        addresses[j.get('name')]=address
        address+=7 if j.tag=='freejoint' or j.get('type')=='free' else 4 if j.get('type')=='ball' else 1
    initial=json.loads((IGRIS/'configs/motions/WALK_STANDBY.json').read_text())['motions'][0]['position']
    initial=np.array(initial,float);initial[OWNED]=z['q_command_rad'][0]
    # Parallel motor joints in the vendor neutral keyframe are zero. Initialize
    # matching serial wrist joints at zero, then approach the recorded start.
    seed_path=campaign/'seed_qpos.npy'
    if not seed_path.exists():initial[[20,21,27,28]]=0.
    for key in root.findall('./keyframe/key'):
        q=list(map(float,key.get('qpos').split()))
        for i,n in enumerate(NAMES):q[addresses[n]]=initial[i]
        if seed_path.exists():
            # A validated, settled native state keeps every parallel coupler
            # consistent. It is a reset pose only; the base remains a free joint.
            q=np.load(seed_path,allow_pickle=False).tolist()
            assert len(q)==address and np.isfinite(q).all()
        key.set('qpos',' '.join(map(str,q)))
    assert root.find('.//joint[@name="floating_base"]') is not None or root.find('.//freejoint[@name="floating_base"]') is not None
    tree.write(out/'scene.xml',encoding='unicode')
    shutil.copytree(IGRIS/'configs/motions',out/'motions')
    for file in ('HOME.json','WALK_STANDBY.json'):
        motion=json.loads((out/'motions'/file).read_text())
        for m in motion['motions']:m['position']=initial.tolist()
        (out/'motions'/file).write_text(json.dumps(motion))
    config=yaml.safe_load((IGRIS/'sim_local/robot.yaml').read_text())
    c=config['igris_c'];c['description']['mujoco']['robot_model_path']=container_path(out/'scene.xml')
    c['controller']['motion_root_dir']=container_path(out/'motions')
    assert c['description']['mujoco']['fixed']=='false' and c['controller']['GravityCompensationModule']['enabled']
    assert c['network']['cyclonedds']['namespace']['user_suffix']=='001'
    (out/'robot.yaml').write_text(yaml.safe_dump(config))
    (out/'trial.json').write_text(json.dumps(dict(name=name,mass_kg=mass,gains=gains,initial_q=initial.tolist(),payload_offset_m=offset,hold_seconds=30),indent=2))
    return out

def trial(out):
    inspect=json.loads(subprocess.check_output(['docker','inspect','igris-sim']))[0]
    if inspect['HostConfig']['NetworkMode']!='bridge' or inspect['HostConfig']['IpcMode']!='private':
        raise RuntimeError('Dedicated bridge-network/private-IPC simulator required')
    if (out/'result.json').exists():raise FileExistsError('Refusing to overwrite a trial')
    # Only experiment-owned launch processes can be stopped here.
    prefix='^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch igris_c_controller simulation.launch.py'
    active=shell(f"pgrep -af '{prefix}' || true",capture_output=True,text=True).stdout
    if any('config_file:=/igris/sim_local/gain_tuning_100hz/results/' not in line for line in active.splitlines()):
        raise RuntimeError('A simulator outside this experiment is running')
    shell(f"pkill -INT -f '{prefix} config_file:=/igris/sim_local/gain_tuning_100hz/results/' || true")
    time.sleep(3)
    p=container_path(out)
    command=f'source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash; cd /tmp; export DISPLAY=:99 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_PRELOAD={container_path(HERE)}/observe.so IGRIS_GAIN_LOG={p}/physics.bin; exec ros2 launch igris_c_controller simulation.launch.py config_file:={p}/robot.yaml > {p}/vendor.log 2>&1'
    subprocess.run(['docker','exec','-d','igris-sim','bash','-lc',command],check=True)
    time.sleep(8)
    worker=f'source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec python3 {container_path(Path(__file__))} --worker {p}'
    try:
        with (out/'worker.log').open('w') as log:
            pworker=subprocess.Popen(['docker','exec','igris-sim','bash','-lc',worker],stdout=log,stderr=subprocess.STDOUT)
            for args in [('sim','resume'),('torque','on'),('mode','home')]:
                subprocess.run([str(IGRIS/'sim_local/ctl.sh'),*args],check=True,stdout=log,stderr=log)
            subprocess.run([str(IGRIS/'sim_local/rt_physics.sh')],check=True,stdout=log,stderr=log)
            time.sleep(2)
            reset=subprocess.Popen([str(IGRIS/'sim_local/ctl.sh'),'sim','reset'],stdout=log,stderr=log)
            time.sleep(.08)
            walk=subprocess.Popen([str(IGRIS/'sim_local/ctl.sh'),'mode','lowwalk'],stdout=log,stderr=log)
            if reset.wait() or walk.wait():raise RuntimeError('Simulation bringup failed')
            code=pworker.wait(timeout=200)
            if code:raise RuntimeError(f'Worker failed ({code}); inspect {out}/worker.log')
    finally:
        subprocess.run([str(IGRIS/'sim_local/ctl.sh'),'sim','pause'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if 'pworker' in locals() and pworker.poll() is None:
            shell(f"pkill -INT -f '^python3 {container_path(Path(__file__))} --worker {p}$' || true")
    print('FINISHED',out,flush=True)

def physics_last(path):
    with path.open('rb') as f:
        nq,nv,nu=struct.unpack('iii',f.read(12));width=2+nq+nv+2*nu+16
        n=(f.seek(0,2)-12)//(8*width)
        if n<1:raise RuntimeError('No physics state')
        f.seek(12+(n-1)*8*width)
        return np.frombuffer(f.read(8*width),float),nq,nv,nu

def worker(out):
    # This path can run only inside the inspected container with its private IPC.
    if not Path('/.dockerenv').exists() or not str(out).startswith('/igris/sim_local/gain_tuning_100hz/results/'):
        raise RuntimeError('Only the dedicated local simulation is allowed')
    sys.path.insert(0,'/igris/igris_curobo/tools')
    from dds_zmq_bridge import _ros_cmd_writer,_state_types,lowstate_to_dict
    from cyclonedds.core import Policy,Qos
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader,Subscriber
    from cyclonedds.topic import Topic
    cfg=json.loads((out/'trial.json').read_text());g=np.array(cfg['gains'],float)
    assert g.shape==(17,2) and np.isfinite(g).all() and np.all((g[:,0]>0)&(g[:,0]<=600)) and np.all((g[:,1]>0)&(g[:,1]<=5))
    z=np.load(out.parent/'motion.npz',allow_pickle=False);t=z['command_t_s'];q=z['q_command_rad']
    qos=Qos(Policy.Reliability.BestEffort,Policy.Durability.Volatile,Policy.History.KeepLast(20))
    participant=DomainParticipant(0);subscriber=Subscriber(participant);st,mt=_state_types('/igris')
    sr=DataReader(subscriber,Topic(participant,'igris_c_001/rt/lowstate',st,qos=qos),qos=qos)
    mr=DataReader(subscriber,Topic(participant,'igris_c_001/rt/robotstate',mt,qos=qos),qos=qos)
    send=_ros_cmd_writer('igris_c_001','/igris')
    cmd=dict(q=[0.]*31,dq=[0.]*31,kp=[0.]*31,kd=[0.]*31,kinematic_modes=[1]*5)
    ready_g=np.array([[250.,5.]]*3+([[250.,5.]]*4+[[150.,5.]]*3)*2)
    def set_gains(values):
        for i,(kp,kd) in zip(OWNED,values):cmd['kp'][i]=float(kp);cmd['kd'][i]=float(kd)
    set_gains(ready_g)
    mode=None;state=None;state_at=-1.;mode_at=-1.;states=[];commands=[];seq=0
    def observe():
        nonlocal mode,state,state_at,mode_at
        now=time.monotonic()
        for m in mr.take(N=30):mode=m.state.name;mode_at=now
        for s in sr.take(N=50):
            state=lowstate_to_dict(s,'igris_c_001',now,mode);state_at=now
            states.append([now,state['seq'],*state['q'],*state['dq'],*state['tau']])
        return now
    def publish(target,phase,index,deadline):
        nonlocal seq
        now=observe()
        for i,v in zip(OWNED,target):cmd['q'][i]=float(v)
        send(cmd,seq);seq+=1
        commands.append([time.monotonic(),deadline,index,phase,*target])
        return now
    result=dict(profile=cfg['name'],mass_kg=cfg['mass_kg'],hardware_executed=False,
        transport='100 Hz ROS LowCmd -> vendor bridge -> native controller; PJS, dq=tau=0',error=None,completed=False)
    try:
        begin=time.monotonic();ready_since=None;initial=np.array(cfg['initial_q'])[OWNED]
        while time.monotonic()-begin<35:
            now=publish(initial,0,-1,time.monotonic())
            if mode=='ROBOT_STATE_WALK_LOW' and state is not None:
                row,nq,nv,nu=physics_last(out/'physics.bin');base=row[2:9];tilt=np.rad2deg(np.arccos(np.clip(1-2*(base[4]**2+base[5]**2),-1,1)))
                good=base[2]>.7 and tilt<8 and np.max(np.abs(np.array(state['q'])[OWNED]-initial))<.15
                if good:
                    if ready_since is None:ready_since=now
                    if now-ready_since>3:break
                else:ready_since=None
            time.sleep(.01)
        else:raise RuntimeError(f'Initial WALK_LOW stabilization failed, mode={mode}')
        # Same finite approach and gain transition for every candidate. Never
        # teleport a loaded parallel linkage into an inconsistent joint state.
        for k in range(1001):
            u=min(k/400.,1.);blend=u*u*u*(10-15*u+6*u*u)
            target=initial+blend*(q[0]-initial)
            gain_u=np.clip((k-500)/200.,0.,1.)
            set_gains(ready_g+gain_u*(g-ready_g))
            publish(target,0,-1,time.monotonic());time.sleep(.01)
        links={}
        for line in (out/'physics.bin.map').read_text().splitlines():
            v=line.split('\t')
            if v[0]=='joint':links[v[1]]=int(v[2])
        def check_mechanism(row):
            offsets=[row[2+links[NAMES[i]+'_backlash']] for i in OWNED]
            if max(abs(v) for v in offsets)>.15:raise RuntimeError('Passive backlash exceeds 0.15 rad: invalid physical linkage')
            for i in (20,21,27,28):
                actual=row[2+links[NAMES[i]]]
                if abs(actual-state['q'][i])>.25:raise RuntimeError('Native parallel link differs from SDK feedback by >0.25 rad')
        check_mechanism(physics_last(out/'physics.bin')[0])
        assert np.allclose(np.array(cmd['kp'])[OWNED],g[:,0]) and np.allclose(np.array(cmd['kd'])[OWNED],g[:,1])
        result['sent_kp_sdk31']=cmd['kp'].copy();result['sent_kd_sdk31']=cmd['kd'].copy()
        start=time.monotonic()+.05;result['replay_start_monotonic']=start
        hold_samples=round(float(cfg.get('hold_seconds',10))*100)
        if not 1000<=hold_samples<=6000:raise ValueError('Hold duration must be 10 to 60 s')
        result['hold_seconds']=hold_samples/100
        for k in range(len(t)+hold_samples):
            deadline=start+k*.01;time.sleep(max(0.,deadline-time.monotonic()))
            now=observe()
            if now-deadline>.05:raise RuntimeError(f'Publisher lateness {now-deadline:.4f} s')
            if now-state_at>.15 or now-mode_at>.3 or mode!='ROBOT_STATE_WALK_LOW':raise RuntimeError('Stale state or lost WALK_LOW')
            target=q[min(k,len(q)-1)]
            if np.max(np.abs(np.array(state['q'])[OWNED]-target))>.5:raise RuntimeError('Tracking error exceeds 0.5 rad')
            if k%10==0:
                row,nq,nv,nu=physics_last(out/'physics.bin');base=row[2:9]
                tilt=np.rad2deg(np.arccos(np.clip(1-2*(base[4]**2+base[5]**2),-1,1)))
                if now-row[0]>.15 or not np.isfinite(row).all() or base[2]<.7 or tilt>15:raise RuntimeError('Stale physics or lost balance')
                if row[-2]>0:raise RuntimeError('Native self-contact above 0.5 N')
                check_mechanism(row)
            publish(target,1 if k<len(q) else 2,k,deadline)
            if k%2000==0:print('REPLAY',k,'/',len(q),'mode',mode,flush=True)
        result['completed']=True
    except BaseException as e:
        result['error']=str(e)
        raise
    finally:
        np.savez_compressed(out/'signals.npz',commands=np.asarray(commands),states=np.asarray(states))
        result['submitted_replay_samples']=sum(r[3]==1 for r in commands)
        result['submitted_hold_samples']=sum(r[3]==2 for r in commands)
        result['end_monotonic']=time.monotonic()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--trial',type=Path);group.add_argument('--worker',type=Path)
    parser.add_argument('--seed',type=Path,help='complete settled native qpos .npy for reproducible --prepare')
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare.resolve(),args.seed)
    elif args.trial:trial(args.trial.resolve())
    else:worker(args.worker.resolve())
