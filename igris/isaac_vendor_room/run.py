"""Private vendor simulator lifecycle and upper-body LowCmd client.

Prepare a new output directory first; this launcher never connects to hardware.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from control_gc import defer_cyclic_gc

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
JOINTS = ['waist_yaw','waist_roll','waist_pitch'] + [f'{s}_{j}' for s in ('l','r') for j in ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll')] + [f'{s}_{j}' for s in ('l','r') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch')] + ['neck_yaw','neck_pitch']
OWNED = [0,1,2,*range(15,31)]
# Opt-in loaded-zero coordinator: single LowCmd writer, measured waist, and
# zero-owned-effort-before-pause cleanup. Normal WALK_LOW stays unchanged.
HTD_LOADED_ZERO_RUNTIME_CONTRACT = 'htd_loaded_zero/v1'


def execute(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def adaptive_clearance_error(response,heartbeat,wall,control_time):
    """Validate retained accepted metadata before it can mutate the lift cap."""
    try:
        values=[response['wall'],heartbeat['wall'],response['reference_control_time_s'],
                response['end_time'],response['position_error_m'],wall,control_time]
        valid=(all(math.isfinite(v) for v in values)
               and response['accepted'] is True and response['prior_status']=='used'
               and 0<=wall-response['wall']<.25 and 0<=wall-heartbeat['wall']<.25
               and 0<=control_time-response['reference_control_time_s']<.1
               and control_time<=response['end_time']+.1 and response['position_error_m']>=0)
    except (KeyError,TypeError,ValueError):
        valid=False
    if not valid:raise RuntimeError('Adaptive clearance requires fresh accepted live HL-IK metadata')
    return response['position_error_m']


def current_adaptive_clearance_error(out,control_time):
    """Read the newest planning result before testing its unmodified age."""
    response=json.loads((out/'live_ik_response.json').read_text())
    heartbeat=json.loads((out/'live_ik_heartbeat.json').read_text())
    return adaptive_clearance_error(response,heartbeat,time.time(),control_time)


def validate_live_solver_selection(manifest,ready):
    """Require the prepared SIM transport solver mode before acquisition."""
    scope=manifest.get('htd_loaded_zero') or {}
    if scope.get('enabled') is not True or scope.get('scope')!='loaded_transport':return None
    source=ROOT/'igris/hl_ik/pipeline.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    budget={'outer':10,'inner':10}
    if (scope.get('transport',{}).get('primary_qp_start')!='cold'
            or scope.get('transport',{}).get('primary_qp_iteration_limits')!=budget
            or ready.get('primary_qp_start')!='cold'
            or ready.get('primary_qp_iteration_limits')!=budget
            or ready.get('pipeline_source_sha256')!=digest
            or scope.get('setup_sha256',{}).get(str(source))!=digest):
        raise RuntimeError('Loaded transport live solver selection or source differs from setup')
    return dict(primary_qp_start='cold',pipeline_source_sha256=digest,primary_qp_iteration_limits=budget)


def publish_lowcmd(publisher,message_factory,command,seq,*,before_publish=None):
    modes=command.get('kinematic_modes',[1]*5)
    if len(modes)!=5 or any(value not in (0,1) for value in modes):
        raise ValueError('Invalid kinematic modes')
    msg=message_factory();msg.header.seq=seq;msg.header.frame_id='';msg.kinematic_modes=[int(value) for value in modes]
    if len(msg.motors)!=31:raise ValueError('Expected31 SDK motor fields')
    for i,motor in enumerate(msg.motors):
        motor.id=i;motor.q=float(command['q'][i]);motor.dq=float(command['dq'][i]);motor.tau=float(command.get('tau',[0.]*31)[i])
        motor.kp=float(command['kp'][i]);motor.kd=float(command['kd'][i])
    if before_publish is not None:before_publish()
    publisher.publish(msg)


def request_replay_pacing(out,rate,motion_phase,start_phase):
    if rate is None or motion_phase<start_phase or (out/'pacing.request').exists():
        return False
    (out/'pacing.request').touch(exist_ok=False)
    return True


def request_room_layout(out,manifest,pelvis_pose,torso_pose=None,*,pose_native_time=None):
    """Stage once after preparation; reject a known bad HTD approach before grasp."""
    import numpy as np
    scope=manifest.get('htd_loaded_zero') or {}
    walking=manifest.get('walking_transfer') or {}
    evidence=None
    if (scope.get('enabled') is True and scope.get('scope')=='loaded_transport'
            and walking.get('heading_preserving_route') is True):
        torso=np.asarray(torso_pose,dtype=float)
        if (torso.shape!=(7,) or not np.isfinite(torso).all()
                or abs(np.linalg.norm(torso[3:])-1)>1e-6):
            raise ValueError('Finite unit-quaternion pre-grasp torso pose required')
        heading=walking['target_torso_heading_rad']
        if pose_native_time is None or not np.isfinite([heading,pose_native_time]).all() or pose_native_time<0:
            raise ValueError('Finite pre-grasp heading and native pose time required')
        w,x,y,z=torso[3:]
        measured=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        difference=float(np.arctan2(np.sin(measured-heading),np.cos(measured-heading)))
        if abs(difference)>.12:
            raise RuntimeError(f'Pre-grasp heading differs from reviewed cart approach: measured={measured:.9f}, '
                               f'reviewed={heading:.9f}, error={difference:.9f}, limit=0.12 rad; room not staged')
        evidence=dict(measured_torso_heading_rad=measured,reviewed_torso_heading_rad=float(heading),
                      heading_error_rad=difference,limit_rad=.12,pose_native_time=float(pose_native_time),
                      stage='prepared upper pose before room staging and grasp',passed=True)
    reference=manifest.get('room_reference_base_xyyaw',[0.,0.,0.])
    w,x,y,z=pelvis_pose[3:];yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))-reference[2]
    c,s=np.cos(yaw),np.sin(yaw)
    tx=pelvis_pose[0]-c*reference[0]+s*reference[1];ty=pelvis_pose[1]-s*reference[0]-c*reference[1]
    # Native staging checks existence before parsing: publish only complete data.
    temporary=out/'room_layout.request.tmp'
    temporary.write_text(f'{tx} {ty} {yaw}\n')
    temporary.replace(out/'room_layout.request')
    return evidence


def loaded_start_environment(out,manifest):
    scope=manifest.get('htd_loaded_zero')
    if scope is not None or 'htd_prealignment_coast' in manifest:
        from htd_loaded import prealignment_coast_selection
        prealignment_coast_selection(manifest)
    upper_timeconst=None
    if scope is not None or 'htd_upper_backlash_limit' in manifest:
        from htd_loaded import upper_backlash_selection
        upper_timeconst=upper_backlash_selection(manifest)
    if (scope is None and not any(key in manifest for key in
            ('dip_damping','dip_damping_requested','dip_damping_scope','dip_damping_descriptor_sha256'))
            and not (out/'dip_damping.json').exists() and not any(out.glob('dip_damping_*'))):return ''
    from dip_damping import validate as validate_dip_damping
    dip_variant=validate_dip_damping(out)
    if dip_variant is not None and (scope is None or scope.get('scope')!='loaded_transport'):
        raise ValueError('DIP damping requires actual loaded_transport scope before native start')
    if scope is None:return ''
    from htd_loaded_setup import require_runtime_hooks
    require_runtime_hooks(sys.modules[__name__])
    loaded_rate=scope.get('loaded_rate')
    if (isinstance(loaded_rate,bool) or not isinstance(loaded_rate,(int,float))
            or not math.isfinite(loaded_rate) or loaded_rate not in (.1,.05)):
        raise ValueError('HTD loaded rate must be exactly0.1 or0.05')
    if loaded_rate==.05 and (scope.get('scope')!='loaded_transport'
            or scope.get('current_grasp_reference') is not True
            or manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'
            or manifest.get('evaluation_profile')!='relaxed_transport'):
        raise ValueError('HTD loaded rate0.05 requires explicit relaxed SIM transport/current-grasp scope')
    request=out/'htd_pacing.request';ack=out/'htd_pacing.request.ack'
    if (scope.get('enabled') is not True or scope.get('schema')!='htd_loaded_zero/v1'
            or 'htd_initial_state' in manifest or manifest.get('simulation_rate_limit')!=.5
            or manifest.get('simulation_rate_start_phase_s')!=90.
            or scope.get('acquisition_rate')!=.5
            or scope.get('pacing_request_path')!=str(request) or scope.get('pacing_ack_path')!=str(ack)
            or request.exists() or ack.exists()
            or any((out/n).exists()for n in ('htd_initial_state.request','htd_initial_state_applied.json'))):
        raise ValueError('Fresh explicit loaded-zero pacing scope required')
    for name,expected in scope.get('setup_sha256',{}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Loaded setup changed before start: '+name)
    from htd_backlash_limits import validate as validate_backlash
    validate_backlash(out, expected_upper_timeconst=upper_timeconst)
    library=scope.get('observer_library')
    if (not library or os.environ.get('IGRIS_ROOM_PROFILE_LIBRARY')!=library
            or scope.get('setup_sha256',{}).get(library)!=hashlib.sha256(Path(library).read_bytes()).hexdigest()):
        raise ValueError('Loaded native observer interposer/hash differs')
    import yaml
    controller=yaml.safe_load((out/'robot.yaml').read_text())['igris_c']['controller']
    if (scope.get('params_path')!=str(out/'htd_gc_params.yaml')
            or controller['gain_parameter_path']!=scope['params_path']
            or controller['GravityCompensationModule']['enabled'] is not True):
        raise ValueError('Loaded trial-local lower15 GC scope differs')
    rate_text='.1' if loaded_rate==.1 else '.05'
    environment='export IGRIS_ROOM_PACER_SLOWDOWN='+shlex.quote(str(request))+' IGRIS_ROOM_SLOW_REALTIME_FACTOR='+rate_text
    if scope.get('command_observer') is not None:
        from htd_command_scope import validate
        env=validate(out,scope['command_observer'])
        environment+='\nexport '+' '.join(key+'='+shlex.quote(value) for key,value in env.items())
    return environment


def validate_payload_geometry(geom,payload_config):
    import numpy as np
    if geom is None:raise ValueError('Payload geometry is missing')
    size=2*np.fromstring(geom.get('size',''),sep=' ')
    mass=float(geom.get('mass','nan'));expected=float(payload_config['mass_kg'])
    # USD Physics MassAPI stores float32. The native export can therefore
    # contain .10000000149 for requested .1; accept that exact roundtrip too.
    mass_matches=(np.isclose(mass,expected,rtol=0,atol=1e-9)
                  or np.isclose(mass,float(np.float32(expected)),rtol=0,atol=1e-9))
    if (size.shape!=(3,) or not np.allclose(size,payload_config['size_m'],rtol=0,atol=1e-6)
        or not np.isfinite([mass,expected]).all() or min(mass,expected)<=0 or not mass_matches):
        raise ValueError('Payload dimensions/mass differ from manifest')


def config(out):
    cfg=json.loads((out/'runtime.json').read_text())
    info=json.loads(subprocess.check_output(['docker','inspect',cfg['container']]))[0]
    if (info['HostConfig']['NetworkMode']!='none' or info['HostConfig']['IpcMode']!='private'
        or info['Config'].get('Labels',{}).get('igris.room.output')!=str(out)):
        raise RuntimeError('Refusing a container outside this isolated simulation')
    return cfg


def shell(out,script,**kwargs):
    cfg=config(out)
    prefix=f'source /opt/ros/jazzy/setup.bash\nsource /igris/ws/install/setup.bash\nsource /igris/sim_local/tools_ws/install/setup.bash\nexport CYCLONEDDS_URI={shlex.quote("file://"+str(out/"cyclonedds.xml"))}\n'
    return execute(['docker','exec','-u',f'{os.getuid()}:{os.getgid()}',cfg['container'],'bash','-lc',prefix+script],**kwargs)


def latest(out):
    import numpy as np
    meta=json.loads((out/'model.json').read_text())
    row=np.frombuffer((out/'latest.bin').read_bytes(),dtype=np.float64)
    if len(row)!=2+7*len(meta['bodies']) or not np.isfinite(row).all() or not 0<=time.time()-row[0]<.5:
        raise RuntimeError('Missing, invalid or stale native physics state')
    poses=row[2:].reshape(-1,7)
    return row[:2],poses,next(i for i,b in enumerate(meta['bodies']) if b['name']=='pelvis')


def ctl(out,*args):
    shell(out,'exec ros2 run igris_sim_ctl igris_ctl '+shlex.join(args)+' --cyclonedds-xml '+shlex.quote(str(out/'cyclonedds.xml')),
          stdout=subprocess.DEVNULL)


def start(out,image,native_draw=False):
    if (out/'runtime.json').exists() or (out/'poses.bin').exists():
        raise FileExistsError('Use a fresh prepared output directory for each run')
    inc=ROOT/'igris/ws/install/igris_c_mujoco/include'
    lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    manifest=json.loads((out/'manifest.json').read_text())
    loaded_environment=loaded_start_environment(out,manifest)
    stage_selector=None
    stage_scope=manifest.get('room_staging',{}).get('scope')
    if stage_scope in ('source_only','source_and_destination'):
        if stage_scope=='source_only':
            from rolltainer_staging import staging_contract
        else:
            from inplace_layout import near_staging_contract as staging_contract
        selector=staging_contract(manifest)
        if selector['transform_bodies']!=manifest['room_staging']['transform_bodies']:raise ValueError('Source staging selector differs from manifest')
        stage_selector=out/'room_stage_bodies.txt'
        stage_selector.write_text('\n'.join(selector['transform_bodies'])+'\n')
    support_files=[]
    if manifest.get('source_support_bodies') and manifest.get('destination_bodies'):
        for kind in ('source','destination'):
            file=out/(kind+'_support_bodies.txt')
            file.write_text('\n'.join(manifest[kind+'_support_bodies'] if kind=='source' else manifest['destination_bodies'])+'\n')
            support_files.append(file)
    if manifest.get('payload'):
        import xml.etree.ElementTree as ET
        payload_config=manifest['payload']
        geom=ET.parse(out/'scene.xml').find(f".//body[@name='{payload_config['body']}']/geom[@name='{payload_config['geom']}']")
        validate_payload_geometry(geom,payload_config)
    payload=manifest.get('payload',{}).get('body','')
    execute(['g++','-O2','-std=c++17',f'-I{inc}',str(HERE/'observer.cpp'),f'-L{lib}',f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-ldl','-o',str(out/'audit')])
    solver_args=[]
    if manifest.get('contact_solver'):
        solver_args=['--noslip-iterations',str(manifest['contact_solver']['noslip_iterations'])]
    if manifest.get('friction_impedance'):
        solver_args+=['--impratio',str(manifest['friction_impedance']['impratio'])]
    execute([str(out/'audit'),str(out/'vendor.xml'),str(out/'scene.xml'),str(out/'model.json')]+solver_args)
    execute(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(HERE/'observer.cpp'),'-ldl','-o',str(out/'observer.so')])
    execute(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(ROOT/'igris/hl_ik/native_clock.cpp'),'-ldl','-o',str(out/'clock.so')])
    preload=[out/'clock.so',out/'observer.so']
    command_observer=manifest.get('htd_loaded_zero',{}).get('command_observer')
    if command_observer is not None:
        preload.insert(0,Path(command_observer['library']))
    native_policy_clock=manifest.get('vendor_policy_native_clock',False)
    if not isinstance(native_policy_clock,bool):raise ValueError('Native policy clock flag must be boolean')
    policy_binary=None
    if native_policy_clock:
        from vendor_policy_clock import build,BINARY
        preload.insert(0,build(out));policy_binary=BINARY.resolve()
    rate=manifest.get('simulation_rate_limit')
    if rate is not None:
        if not math.isfinite(rate) or not 0<rate<=1:raise ValueError('Invalid simulation wall-rate limit')
        execute(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(HERE/'simulation_pacer.cpp'),'-ldl','-o',str(out/'pacer.so')])
    profile=os.environ.get('IGRIS_ROOM_PROFILE_LIBRARY')
    if profile:preload.insert(0,Path(profile).resolve())
    if manifest.get('compliant_hands'):
        from hands import MOTOR
        execute(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(MOTOR),'-ldl','-o',str(out/'hand_motor.so')])
        preload.insert(1,out/'hand_motor.so')
    if manifest.get('stage_room_after_preparation'):
        execute(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(HERE/'room_setup.cpp'),'-ldl','-o',str(out/'room_setup.so')])
        preload.insert(1,out/'room_setup.so')
    initial_state=None
    if manifest.get('htd_initial_state'):
        from htd_initial_state import validate as validate_initial_state
        initial_state=validate_initial_state(out,manifest)
        if any((out/n).exists()for n in ('htd_initial_state.request','htd_initial_state_applied.json')):
            raise ValueError('Initial-state activation files predate startup')
        preload.insert(0,Path(initial_state['library']))
    if rate is not None:preload.insert(0,out/'pacer.so')
    name='igris-room-'+hashlib.sha256(str(out).encode()).hexdigest()[:10]
    (out/'vendor_logs').mkdir()
    execute(['docker','run','-d','--name',name,'--label',f'igris.room.output={out}','--network','none','--ipc','private','--shm-size','2g',
        '--ulimit','memlock=-1','--ulimit','rtprio=99','--cap-add','SYS_NICE','-e','DISPLAY=:99','-e','HOME=/tmp','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
        '-v',f'{ROOT}:{ROOT}:ro','-v',f'{ROOT}/igris:/igris:ro','-v',f'{ROOT}/igris/configs:/robros/configs:ro','-v',f'{out}:{out}','-v',f'{out}/vendor_logs:/robros/logs',
        image,'sleep','infinity'],stdout=subprocess.DEVNULL)
    (out/'runtime.json').write_text(json.dumps({'container':name,'image':image,'network':'none','ipc':'private','native_draw':native_draw,'simulation_rate_limit':rate,'vendor_policy_native_clock':native_policy_clock,'hardware_executed':False},indent=2))
    execute(['docker','exec','-d',name,'Xvfb',':99','-screen','0','800x600x24'])
    script=f'''source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
cd /tmp
export LD_PRELOAD={shlex.quote(':'.join(map(str,preload)))} IGRIS_ROOM_OUTPUT={shlex.quote(str(out))}
export IGRIS_HAND_DIR={shlex.quote(str(out))} IGRIS_HAND_LOG_WALL=1
export IGRIS_HLIK_CLOCK={shlex.quote(str(out/'native_clock.bin'))}
{'export IGRIS_ROOM_POLICY_CLOCK_BINARY='+shlex.quote(str(policy_binary)) if native_policy_clock else ''}
{'export IGRIS_ROOM_MAX_REALTIME_FACTOR='+str(rate) if rate is not None else ''}
{'export IGRIS_ROOM_PACER_START='+shlex.quote(str(out/'pacing.request')) if rate is not None else ''}
{loaded_environment}
export IGRIS_ROOM_PAYLOAD={shlex.quote(payload)}
{'export IGRIS_HTD_INITIAL_STATE_DIR='+shlex.quote(str(out)) if initial_state else ''}
{'export IGRIS_ROOM_STAGE='+shlex.quote(str(out)) if manifest.get('stage_room_after_preparation') else ''}
{'export IGRIS_ROOM_STAGE_BODIES_FILE='+shlex.quote(str(stage_selector)) if stage_selector else ''}
{'export IGRIS_ROOM_STAGE_SCOPE='+shlex.quote(stage_scope) if stage_selector else ''}
{'export IGRIS_ROOM_SOURCE_SUPPORT_BODIES_FILE='+shlex.quote(str(support_files[0]))+' IGRIS_ROOM_DESTINATION_SUPPORT_BODIES_FILE='+shlex.quote(str(support_files[1])) if support_files else ''}
{'export IGRIS_ROOM_SKIP_NATIVE_DRAW=1' if not native_draw else ''}
exec ros2 launch igris_c_controller simulation.launch.py config_file:={shlex.quote(str(out/'robot.yaml'))} > {shlex.quote(str(out/'vendor.log'))} 2>&1'''
    execute(['docker','exec','-d','-u',f'{os.getuid()}:{os.getgid()}',name,'bash','-lc',script])
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        log=(out/'vendor.log').read_text(errors='replace') if (out/'vendor.log').exists() else ''
        if 'MujocoRos node initialized, joint states extracted and published' in log and 'Control:' in log:
            break
        if 'aborting launch' in log:
            raise RuntimeError('Vendor launch failed; inspect vendor.log')
        time.sleep(.2)
    else:raise RuntimeError('Vendor readiness timed out; inspect vendor.log')
    print('STARTED',name,flush=True)


def bringup(out,cpu=None):
    if cpu is None:
        allowed=os.sched_getaffinity(0)
        def speed(i):
            p=Path(f'/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq')
            return int(p.read_text()) if p.exists() else 0
        cpu=max(sorted(allowed),key=speed)
    if cpu not in os.sched_getaffinity(0):raise ValueError('Physics CPU is not available')
    # WALK_LOW owns the legs; its upper-body input must already have valid
    # gains when the mode begins. Otherwise the massive hands can fold the
    # unpowered wrists during DDS/worker initialization.
    shell(out,'python3 - '+shlex.quote(str(out))+' '+shlex.quote(str(HERE/'upper_hold.py'))+''' <<'INNER'
from pathlib import Path
import subprocess,sys
out=Path(sys.argv[1])
with (out/'upper_hold.log').open('w') as log:
    subprocess.Popen(['python3',sys.argv[2],'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
INNER''')
    deadline=time.monotonic()+15
    while not (out/'upper_hold.ready').exists():
        if (out/'stop.request').exists() or time.monotonic()>deadline:
            raise RuntimeError('Upper-body initialization hold failed; see upper_hold.log')
        time.sleep(.02)
    ctl(out,'sim','reset');ctl(out,'sim','resume');ctl(out,'torque','on');ctl(out,'mode','home')
    # Same native scheduling helper as the validated vendor experiments, scoped
    # to this network/IPC-isolated container. It changes no physics parameters.
    shell(out,'python3 - '+str(cpu)+' '+shlex.quote(str(out/'physics_stats.json'))+''' <<'INNER'
from pathlib import Path
import json,os,subprocess,sys,time
p=Path(sys.argv[2]);deadline=time.monotonic()+5
while not p.exists() and time.monotonic()<deadline:time.sleep(.1)
stats=json.loads(p.read_text());tid=stats['tid']
process=Path('/proc')/str(stats['pid'])
if (process/'comm').read_text().strip()!='igris_c_mujoco' or not (process/'task'/str(tid)).exists():raise RuntimeError('Observer physics thread no longer exists')
cpu=int(sys.argv[1]);os.sched_setaffinity(tid,{cpu})
subprocess.run(['chrt','-f','-p','50',str(tid)],check=True)
assert os.sched_getaffinity(tid)=={cpu}
print('PHYSICS_SCHEDULING',tid,'CPU',cpu,'FIFO',50,flush=True)
INNER''')
    cfg=config(out);cfg['physics_cpu']=cpu
    (out/'runtime.json').write_text(json.dumps(cfg,indent=2))
    time.sleep(2)
    for attempt in range(4):
        # Reuse the source experiment's 80 ms launch stagger. Each helper has
        # 800 ms DDS discovery; serial waits leave HOME unsupported for ~1 s.
        shell(out,'python3 - '+shlex.quote(str(out/'cyclonedds.xml'))+''' <<'INNER'
import subprocess,sys,time
base=['ros2','run','igris_sim_ctl','igris_ctl']
tail=['--cyclonedds-xml',sys.argv[1]]
reset=subprocess.Popen(base+['sim','reset']+tail,stdout=subprocess.DEVNULL)
time.sleep(.08)
walk=subprocess.Popen(base+['mode','lowwalk']+tail,stdout=subprocess.DEVNULL)
codes=[reset.wait(),walk.wait()]
if any(codes):raise RuntimeError('Reset/WALK_LOW transition failed')
INNER''')
        time.sleep(5)
        _,poses,base=latest(out)
        q=poses[base,3:]
        tilt=__import__('math').acos(max(-1,min(1,1-2*(q[1]**2+q[2]**2))))
        print('BRINGUP',attempt,'pelvis',poses[base].tolist(),'tilt',tilt,flush=True)
        if poses[base,2]>.7 and tilt<.2:
            if json.loads((out/'manifest.json').read_text()).get('compliant_hands'):
                from hands import CONTROL,PYTHON
                env=os.environ.copy()
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
                env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,ROOT/'igris/igris_curobo/src',ROOT/'igris-core/igris_core',ROOT/'igris/real_client']))
                with (out/'hand_controller.log').open('w') as log:
                    process=subprocess.Popen([str(PYTHON),str(CONTROL),'--run',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                sibling_path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
                siblings={cpu}
                if sibling_path.exists():
                    for part in sibling_path.read_text().strip().split(','):
                        ends=list(map(int,part.split('-')))
                        siblings.update(range(ends[0],ends[-1]+1))
                available=os.sched_getaffinity(0)-siblings
                hand_cpu=min(available,key=lambda n:abs(n-6)) if available else cpu
                os.sched_setaffinity(process.pid,{hand_cpu});cfg['hand_controller_cpu']=hand_cpu
                cfg['hand_controller_pid']=process.pid
                (out/'runtime.json').write_text(json.dumps(cfg,indent=2))
                deadline=time.monotonic()+15
                while not (out/'hand_reference.txt').exists():
                    if process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('MCP controller did not start; see hand_controller.log')
                    time.sleep(.02)
            return
    raise RuntimeError('WALK_LOW standing failed; inspect vendor.log and poses.bin')


def command(out,targets,seconds,motion=None,gains=None,compensate_backlash=False,live_ik=False,walk_to_rolltainer=False,inplace_transfer=False):
    if walk_to_rolltainer and inplace_transfer:raise ValueError('Select only one transfer controller')
    if not Path('/.dockerenv').exists():
        args=[str(HERE/'run.py'),'worker','--out',str(out),'--q',json.dumps(targets),'--seconds',str(seconds)]
        if motion:args+=['--motion',str(motion)]
        if gains:args+=['--gains',str(gains)]
        if compensate_backlash:args+=['--compensate-backlash']
        if live_ik:args+=['--live-ik']
        if walk_to_rolltainer:args+=['--walk-to-rolltainer']
        if inplace_transfer:args+=['--inplace-transfer']
        pythonpath='PYTHONPATH='+shlex.quote(str(ROOT/'env_isaaclab/lib/python3.12/site-packages'))+':$PYTHONPATH ' if walk_to_rolltainer or inplace_transfer else ''
        shell(out,pythonpath+'exec python3 '+shlex.join(args))
        return
    import numpy as np
    import xml.etree.ElementTree as ET
    sys.path.insert(0,str(ROOT/'igris/igris_curobo/tools'))
    from dds_zmq_bridge import lowstate_to_dict,read_joint_offsets,_VENDOR_KP,_VENDOR_KD
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd,LowState,RobotState
    walking=None;walking_state=None;pickup_box_z=None
    room_manifest=json.loads((out/'manifest.json').read_text())
    loaded_scope=room_manifest.get('htd_loaded_zero')
    if loaded_scope is not None:
        from htd_loaded_setup import require_runtime_hooks
        require_runtime_hooks(sys.modules[__name__])
        if not walk_to_rolltainer or inplace_transfer or not live_ik or not compensate_backlash:
            raise ValueError('Loaded zero requires the sole live HL-IK walking publisher')
    if walk_to_rolltainer:
        if not live_ik or room_manifest.get('room_staging',{}).get('scope')!='source_only':
            raise ValueError('Loaded walking requires live HL-IK and a fixed rolltainer destination')
        from walking_transfer import WalkingTransfer
        from igris_c_sdk.msg import WalkTwist
        walking_spec=room_manifest['walking_transfer']
        walking=WalkingTransfer(walking_spec['target_payload_xyz'],walking_spec['target_torso_heading_rad'],
                                room_manifest['payload']['mass_kg'],room_manifest['payload']['friction'],
                                clearance_lift=walking_spec.get('clearance_lift_m',0.),
                                torso_relative_lowering=walking_spec.get('torso_relative_lowering',False),
                                world_position_lowering=walking_spec.get('world_position_lowering',False),
                                heading_preserving_route=walking_spec.get('heading_preserving_route',False),
                                xy_first_lowering=walking_spec.get('xy_first_lowering',False),
                                arrival_settle_seconds=walking_spec.get('arrival_settle_seconds',1.5),
                                clearance_minimum_rise=walking_spec.get('clearance_minimum_rise_m'),
                                stationary_only=loaded_scope is not None and loaded_scope.get('transport') is None,
                                adaptive_clearance=loaded_scope is not None and loaded_scope.get('transport') is not None,
                                arrival_brake_radius=walking_spec.get('arrival_brake_radius_m'))
    if inplace_transfer:
        if not live_ik or room_manifest.get('room_staging',{}).get('scope')!='source_and_destination':
            raise ValueError('In-place transfer requires live HL-IK and shared source/destination staging')
        from inplace_transfer import InplaceTransfer
        from inplace_reference import apply_waist_reference
        from igris_c_sdk.msg import WalkTwist
        walking_spec=room_manifest['inplace_transfer']
        walking=InplaceTransfer(walking_spec['target_scene_xyz'],room_manifest['payload']['mass_kg'],
            room_manifest['payload']['friction'],waist_end_yaw=walking_spec['waist_yaw_end_rad'])
    archived=None
    if live_ik:
        from reference_spline import evaluate as spline_reference
        if not motion or not compensate_backlash:raise ValueError('Live HL-IK requires an archive and native backlash tracking')
        ready=json.loads((out/'live_ik_ready.json').read_text())
        if ready['motion_sha256']!=hashlib.sha256(motion.read_bytes()).hexdigest():raise ValueError('Live HL-IK plan differs from publisher reference')
        validate_live_solver_selection(room_manifest,ready)
        body_names=[b['name'] for b in json.loads((out/'model.json').read_text())['bodies']]
        torso_index=body_names.index('torso');live_started=None
        actual_hand_indices=[body_names.index('end_effector_left'),body_names.index('end_effector_right')]
        payload_index=body_names.index(json.loads((out/'manifest.json').read_text())['payload']['body'])
    if motion:
        from box_replay import load_motion
        stamps,archived,indices=load_motion(motion)
        targets={JOINTS[i]:float(q) for i,q in zip(indices,archived[0])}
        if gains is None:raise ValueError('Replay requires its saved --gains file')
    if not isinstance(targets,dict) or not targets or not np.isfinite(seconds) or seconds<4:
        raise ValueError('Provide named upper-body q targets and at least 4 seconds')
    urdf=ET.parse(ROOT/'igris/igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf')
    if compensate_backlash:
        if not motion:raise ValueError('Backlash compensation requires a validated motion reference')
        sys.path.insert(0,str(ROOT/'igris/igris_curobo/src'))
        from igris_ik.vendor_limits import coupled_halfspaces
        from compensation import compensate
        a,b=coupled_halfspaces([JOINTS[i] for i in indices])
        limits=[urdf.find(f".//joint[@name='{JOINTS[i]}']/limit") for i in indices]
        lo=np.array([float(v.get('lower')) for v in limits]);hi=np.array([float(v.get('upper')) for v in limits])
        filtered_offsets=np.zeros(len(indices));previous_motor=None;last_offsets=None
    for name,value in targets.items():
        if name not in JOINTS or JOINTS.index(name) not in OWNED or not isinstance(value,(float,int)) or not np.isfinite(value):
            raise ValueError(f'Invalid upper-body target: {name}')
        joint=urdf.find(f".//joint[@name='{name}']")
        limit=joint.find('limit') if joint is not None else None
        values=archived[:,list(indices).index(JOINTS.index(name))] if motion else np.asarray([value])
        if limit is None or np.any(values<float(limit.get('lower'))) or np.any(values>float(limit.get('upper'))):
            raise ValueError(f'Joint target outside URDF limits: {name}')
    rclpy.init();node=rclpy.create_node('isaac_room_upper_body')
    state=None;mode=None;numeric_mode=None;environment=None;received=0.;mode_at=0.
    def on_mode(sample):
        nonlocal mode,mode_at,numeric_mode,environment
        numeric_mode=int(sample.state);environment=int(sample.environment)
        mode='ROBOT_STATE_WALK_LOW' if sample.state==RobotState.ROBOT_STATE_WALK_LOW else str(sample.state)
        if sample.environment!=RobotState.ROBOT_ENV_SIM:raise RuntimeError('Simulation state required')
        mode_at=time.monotonic()
    def on_state(sample):
        nonlocal state,received
        state=lowstate_to_dict(sample,'igris_c_001',time.time(),mode);received=time.monotonic()
        state['status']=[int(j.status_bits)for j in sample.joint_state]
    node.create_subscription(LowState,'igris_c_001/rt/lowstate',on_state,qos_profile_sensor_data)
    node.create_subscription(RobotState,'igris_c_001/rt/robotstate',on_mode,qos_profile_sensor_data)
    publisher=node.create_publisher(LowCmd,'igris_c_001/rt/lowcmd',qos_profile_sensor_data)
    loaded=None
    # The vendor velocity subscriber requires RELIABLE, unlike LowState's
    # sensor stream. A best-effort publisher is discovered but never delivered.
    if walking is not None:walk_publisher=node.create_publisher(WalkTwist,'igris_c_001/rt/cmd_vel',10)
    def refresh():
        for _ in range(4):rclpy.spin_once(node,timeout_sec=0)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        refresh()
        if state and mode=='ROBOT_STATE_WALK_LOW':break
        time.sleep(.01)
    else:raise RuntimeError(f'Fresh WALK_LOW required; mode={mode}')
    if loaded_scope is not None:
        sys.path.insert(0,str(ROOT))
        if 5 not in os.sched_getaffinity(0):raise RuntimeError('Reviewed loaded CPU5 worker is unavailable')
        os.sched_setaffinity(0,{5})
        from htd_loaded_runtime import LoadedRuntime
        loaded=LoadedRuntime(out,node)
        refresh()
        loaded.note_environment(environment)
        loaded.warm_observer()
    if walking is not None:
        deadline=time.monotonic()+3
        while not walk_publisher.get_subscription_count():
            refresh()
            if time.monotonic()>deadline:raise RuntimeError('No compatible vendor walking-velocity subscriber')
            time.sleep(.01)
    if (out/'upper_hold.ready').exists():
        (out/'upper_hold.stop').write_text('replay publisher taking ownership')
        deadline=time.monotonic()+2
        while not (out/'upper_hold.stopped').exists():
            if time.monotonic()>deadline:raise RuntimeError('Upper hold did not release command ownership')
            time.sleep(.005)
        hold=json.loads((out/'upper_hold_result.json').read_text())
        if hold['error']:raise RuntimeError('Upper hold failed: '+hold['error'])
    last_sent_command=None
    def send(command,seq,*,before_publish=None):
        nonlocal last_sent_command
        publish_lowcmd(publisher,LowCmd,command,seq,before_publish=before_publish)
        last_sent_command={key:np.array(command.get(key,np.zeros(31)),dtype=float,copy=True)
                           for key in ('q','dq','kp','kd','tau')}
        if 'kinematic_modes' in command:
            last_sent_command['kinematic_modes']=np.array(command['kinematic_modes'],dtype=int,copy=True)
    initial=np.array(state['q'])
    staged_room=json.loads((out/'manifest.json').read_text()).get('stage_room_after_preparation',False)
    if initial.shape!=(31,) or not np.isfinite(initial).all():raise RuntimeError('Invalid measured joint state')
    goal=initial.copy()
    for name,value in targets.items():goal[JOINTS.index(name)]=value
    duration=max(3.,float(np.max(abs(goal-initial)))*1.875/.3)
    if motion:
        spread=initial.copy();spread[16]=.9;spread[23]=-.9
        folded=spread.copy();folded[18]=goal[18];folded[25]=goal[25]
        approach=[initial,spread,folded,goal]
        approach_times=np.r_[0.,np.cumsum([max(2.,float(np.max(abs(b-a)))*1.875/.3) for a,b in zip(approach[:-1],approach[1:])])]
        duration=float(approach_times[-1])
    if not motion and duration>seconds-1:raise ValueError(f'Increase --seconds to at least {duration+1:.2f}')
    kp=np.zeros(31);kd=np.zeros(31)
    kp[OWNED]=np.asarray(_VENDOR_KP)[OWNED];kd[OWNED]=np.minimum(np.asarray(_VENDOR_KD)[OWNED],5.)
    kd[:3]=np.asarray(_VENDOR_KD)[:3]
    kp[29:]=30.;kd[29:]=3.
    if motion:
        import tomllib
        profile=tomllib.loads(gains.read_text())['motion']['gains']
        kp[:]=0.;kd[:]=0.
        for i in indices:
            name=JOINTS[i].replace('l_','left_',1) if JOINTS[i].startswith('l_') else JOINTS[i].replace('r_','right_',1) if JOINTS[i].startswith('r_') else JOINTS[i]
            kp[i]=profile[name]['kp'];kd[i]=profile[name]['kd']
        if not np.isfinite([kp,kd]).all() or np.any(kp<0) or np.any(kd<0) or np.any(kd>5):raise ValueError('Invalid saved replay gains')
        sys.path.insert(0,str(ROOT))
        from igris.hl_ik.native_clock import read_clock
        native_start=read_clock(out/'native_clock.bin')['sim']
        replay_start=None;index=0;seen=set();last_sample_start=None
        live_motion_time=0.;last_control_time=0.;live_segments=[];live_response=None
    result={'pass':False,'targets':targets,'duration_s':seconds,'ramp_s':duration,'hardware_executed':False,'samples':[]}
    result['live_hl_ik']=live_ik
    result['walking_transfer']=walk_to_rolltainer
    result['inplace_transfer']=inplace_transfer
    result['cyclic_gc_deferred_during_commands']=True
    wall_timeout=max(600.,3.*float(stamps[-1])+180.) if motion else seconds
    if walking is not None:wall_timeout=max(wall_timeout,1200.)
    pacing_manifest=json.loads((out/'manifest.json').read_text())
    rate=pacing_manifest.get('simulation_rate_limit')
    pacing_start_phase=pacing_manifest.get('simulation_rate_start_phase_s',0.)
    if not math.isfinite(pacing_start_phase) or pacing_start_phase<0:
        raise ValueError('Invalid pacing start phase')
    if motion and rate is not None:
        wall_timeout=max(wall_timeout,(float(stamps[-1])+duration+120.)/rate+180.)
        result['simulation_rate_limit']=rate
        result['simulation_rate_start_phase_s']=pacing_start_phase
    if loaded_scope is not None and loaded_scope.get('transport') is not None:
        wall_timeout=max(wall_timeout,600.+loaded_scope['duration_s']/loaded_scope['loaded_rate'])
    if motion:
        result.pop('duration_s')
        result.update(motion=str(motion),motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),gains=str(gains),motion_clock='native simulation time; exact archived samples, repeated at 100 Hz wall publisher',motion_duration_s=float(stamps[-1]),archive_samples=len(stamps),wall_timeout_s=wall_timeout)
        result['backlash_compensation_active']=compensate_backlash
        if compensate_backlash:result['motion_clock']='native simulation time; archived output-angle reference with measured backlash compensation'
    output=out/f'command_{time.time_ns()}.json'
    start=time.monotonic();seq=0;loaded_completed=False
    def finish_commands():
        if loaded is not None and last_sent_command is not None:
            loaded.note_environment(environment)
            reason=str(sys.exc_info()[1]) if sys.exc_info()[1] is not None else (
                'bounded loaded diagnostic completed' if loaded_completed else 'command loop ended before loaded completion')
            result['htd_loaded_cleanup']=loaded.pause_cleanup(reason,last_sent_command,lambda c:send(c,seq))
        if live_ik:
            inactive=out/'live_ik_request.inactive.tmp'
            inactive.write_text(json.dumps(dict(active=False,wall=time.time())))
            inactive.replace(out/'live_ik_request.json')
        if walking is not None:
            for _ in range(3):walk_publisher.publish(WalkTwist())
    try:
        with defer_cyclic_gc(before_restore=finish_commands):
            while (age:=time.monotonic()-start)<wall_timeout:
                if (out/'stop.request').exists():raise RuntimeError((out/'stop.request').read_text())
                hand_wall=None
                if (out/'compliance.json').exists():
                    reference=(out/'hand_reference.txt').read_text().split()
                    hand_wall=float(reference[0])
                    if not 0<=time.time()-hand_wall<.25:raise RuntimeError('Stale MCP compliance reference')
                refresh()
                mode_ok=(mode=='ROBOT_STATE_WALK_LOW' if loaded is None or loaded.coordinator.state=='prepared'
                         else numeric_mode in (3,6) if loaded.coordinator.state=='low_pending' else numeric_mode==3)
                if not mode_ok or time.monotonic()-received>.25 or time.monotonic()-mode_at>.5:
                    raise RuntimeError(f'Stale state or WALK_LOW lost: {mode}')
                stamp,poses,base=latest(out)
                if poses[base,2]<.7:raise RuntimeError('Robot fell below 0.7 m')
                progress=read_clock(out/'native_clock.bin')['sim']-native_start if motion else age
                u=min(1.,progress/duration);blend=u**3*(10-15*u+6*u*u)
                target=initial+blend*(goal-initial)
                if motion and progress<duration:
                    phase=min(2,int(np.searchsorted(approach_times,progress,side='right')-1))
                    u=float(np.clip((progress-approach_times[phase])/(approach_times[phase+1]-approach_times[phase]),0,1))
                    target=approach[phase]+u**3*(10-15*u+6*u*u)*(approach[phase+1]-approach[phase])
                sample_index=-1;motion_time=-1.;control_time=-1.;phase_rate=1.
                if staged_room and progress>=duration and not (out/'room_layout.request').exists():
                    heading_check=request_room_layout(out,json.loads((out/'manifest.json').read_text()),
                        poses[base],poses[torso_index] if live_ik else None,pose_native_time=stamp[1])
                    if heading_check is not None:result['pregrasp_heading_check']=heading_check
                if motion and progress>=duration+2.:
                    if staged_room and not (out/'room_layout_applied.json').exists():raise RuntimeError('Room initialization was not acknowledged')
                    if replay_start is None:
                        replay_start=progress;result['replay_start_wall']=time.time();result['replay_start_sim']=native_start+progress
                        print('REPLAY_STARTED',result['replay_start_sim'],len(stamps),'reference samples; backlash correction',compensate_backlash,flush=True)
                    control_time=progress-replay_start
                    motion_time=control_time
                    if live_ik:
                        if live_response is not None:
                            ratio=max(live_response['position_error_m']/.003,
                                      live_response['orientation_error_rad']/np.deg2rad(1.))
                            phase_rate=float(np.clip(2.-ratio,0.,1.))
                        live_motion_time+=max(0.,control_time-last_control_time)*phase_rate
                        if walking is not None:live_motion_time=min(live_motion_time,walking_spec['pickup_stop_phase_s'])
                        if live_response is not None and live_response.get('grasp_gate_time') is not None and not live_response.get('grasp_verified',False):
                            live_motion_time=min(live_motion_time,live_response['grasp_gate_time'])
                        last_control_time=control_time;motion_time=live_motion_time
                    if request_replay_pacing(out,rate,motion_time,pacing_start_phase):
                        result['pacing_activated_control_s']=control_time
                        result['pacing_activated_phase_s']=motion_time
                    # Never skip an archived reference; optional motor correction is logged separately.
                    if index+1<len(stamps) and motion_time>=stamps[index+1]:index+=1
                    sample_index=index;target[indices]=archived[index]
                    if walking is not None:
                        if pickup_box_z is None:pickup_box_z=float(poses[payload_index,2])
                        if walking.started is None and motion_time>=walking_spec['pickup_stop_phase_s']-1e-6:
                            if not live_response or not live_response.get('grasp_verified'):raise RuntimeError('Walking before qualified grasp')
                            forces=json.loads((out/'grasp_force.json').read_text())
                            if not 0<=time.time()-forces['wall']<.25:raise RuntimeError('Stale grasp force before walking')
                            if (poses[payload_index,2]<pickup_box_z+.02 or not np.all(np.asarray(forces['palm_tip_force_N'])[[0,2]]>=walking.required)):
                                raise RuntimeError('Walking requires a raised box and both supporting palms')
                            if inplace_transfer:
                                from scipy.spatial.transform import Rotation
                                placement=json.loads((out/'room_layout_applied.json').read_text())
                                if placement['scope']!='source_and_destination' or not placement['transforms_destination']:
                                    raise RuntimeError('Native near destination was not aligned with source')
                                aligned=Rotation.from_euler('z',placement['yaw_rad']).apply(walking_spec['target_scene_xyz'])
                                aligned[:2]+=placement['translation_xy'];walking.target=aligned
                                result['inplace_target_payload_xyz']=aligned.tolist()
                                walking.start(control_time,poses[base],poses[torso_index],poses[payload_index],
                                    live_response['torso_relative_hand_targets'],target[:3],forces['palm_tip_force_N'],pickup_qualified=True)
                            else:
                                walking.start(control_time,poses[base],poses[torso_index],poses[payload_index],live_response['world_hand_targets'],
                                              torso_relative_hands=live_response['torso_relative_hand_targets'])
                        if walking.started is not None:
                            forces=json.loads((out/'grasp_force.json').read_text())
                            if not 0<=time.time()-forces['wall']<.25:raise RuntimeError('Stale payload support during walking')
                            if forces.get('support_force_groups_configured') is not True:
                                raise RuntimeError('Native source/destination support selectors are not configured')
                            if inplace_transfer:
                                measured_waist=read_joint_offsets(out/'joint_offsets.bin',time.time()) or last_offsets
                                if measured_waist is None or not 0<=time.time()-measured_waist['joint_offsets_t_wall']<.15:
                                    raise RuntimeError('Stale native waist output measurement')
                                waist=np.asarray(state['q'])[:3]+np.asarray(measured_waist['joint_offsets'])[:3]
                                walking_state=walking.update(control_time,poses[base],poses[torso_index],poses[payload_index],waist,
                                    forces['palm_tip_force_N'],forces['source_support_force_world_z_N'],forces['destination_support_force_world_z_N'])
                                target=apply_waist_reference(target,walking_state)
                            else:
                                support_key=('source_support_force_world_z_N' if walking.phase=='clearance'
                                             else 'destination_support_force_world_z_N')
                                external_navigation=(loaded.navigation_signal(control_time,stamp,poses,base,torso_index)
                                    if loaded is not None and loaded.coordinator.transport and walking.phase!='clearance' else None)
                                clearance_error=None
                                if walking.adaptive_clearance and walking.phase=='clearance':
                                    clearance_error=current_adaptive_clearance_error(out,control_time)
                                walking_state=walking.update(control_time,poses[base],poses[torso_index],poses[payload_index],
                                    forces['palm_tip_force_N'],forces[support_key],
                                    hold_carry=loaded is not None and external_navigation is None,
                                    combined_carry_contact=loaded is not None and loaded.coordinator.goal_hold is not None,
                                    external_navigation=external_navigation,
                                    ik_position_error_m=clearance_error)
                            if walking_state['done'] and not (loaded is not None and loaded.coordinator.transport):break
                    elif index==len(stamps)-1:
                        if last_sample_start is None:last_sample_start=control_time
                        elif control_time-last_sample_start>=3.:break
                # Unowned axes have zero position, velocity, torque and gains.
                nominal_target=target.copy();live_active=False
                q=np.zeros(31);controlled=indices if motion else OWNED;q[controlled]=target[controlled]
                offsets=None
                if compensate_backlash:
                    measured=read_joint_offsets(out/'joint_offsets.bin',time.time())
                    if measured is not None:last_offsets=measured
                    # A 30 Hz writer may append just after the reader's timestamp;
                    # keep the preceding sample only while its original stamp is fresh.
                    measured=last_offsets
                    if measured is None or not 0<=time.time()-measured['joint_offsets_t_wall']<.15:raise RuntimeError('Stale native backlash state')
                    offsets=np.asarray(measured['joint_offsets']);filtered_offsets+=.01/(.15+.01)*(offsets[indices]-filtered_offsets)
                    if live_ik:
                        loaded_waist=None
                        actual_measured=np.asarray(state['q'])+offsets
                        if loaded is not None and loaded.coordinator.epoch>0:
                            loaded_waist,actual_measured=loaded.waist_request()
                            nominal_target[:3]=actual_measured[:3];target[:3]=actual_measured[:3]
                        request=dict(wall=time.time(),active=sample_index>=0,motion_time=motion_time,control_time=control_time,
                            nominal=nominal_target.tolist(),measured=actual_measured.tolist(),
                            torso_pose=poses[torso_index].tolist(),payload_pose=poses[payload_index].tolist(),
                            native_pose_sim=float(stamp[1]))
                        if loaded_waist is not None:request['htd_loaded_waist']=loaded_waist
                        if walking_state is not None:request['walking']=walking_state
                        temporary=out/'live_ik_request.tmp';temporary.write_text(json.dumps(request,allow_nan=False))
                        temporary.replace(out/'live_ik_request.json')
                        if sample_index>=0:
                            if live_started is None:live_started=time.monotonic()
                            response_path=out/'live_ik_response.json'
                            if response_path.exists():
                                heartbeat=json.loads((out/'live_ik_heartbeat.json').read_text())
                                if not 0<=time.time()-heartbeat['wall']<.25:raise RuntimeError('Stale live HL-IK worker heartbeat')
                                response=json.loads(response_path.read_text())
                                if control_time>response['end_time']+.10:raise RuntimeError('Live HL-IK reference is more than 100 ms behind simulation')
                                if not response['accepted'] or response['prior_status']!='used':raise RuntimeError('Unvalidated live HL-IK response')
                                from segment_delivery import enqueue_segments
                                live_segments,live_response=enqueue_segments(live_segments,live_response,response,control_time)
                                while len(live_segments)>1 and control_time>=live_segments[1]['start_time']:live_segments.pop(0)
                                segment=live_segments[0]
                                previous=np.asarray(segment['previous']);following=np.asarray(segment['target'])
                                u=float(np.clip((control_time-segment['start_time'])/(segment['end_time']-segment['start_time']),0,1))
                                target[15:29]=spline_reference(segment['control_points'],u);live_active=True
                                if np.any(target[indices]<lo) or np.any(target[indices]>hi) or np.any(a@target[indices]>b+1e-8):raise RuntimeError('Live reference limits')
                            elif time.monotonic()-live_started>1.:raise RuntimeError('Live HL-IK did not respond')
                    if sample_index>=0:
                        blend=float(np.clip(motion_time/2.,0,1));blend=blend**3*(10-15*blend+6*blend*blend)
                        q[indices]=compensate(target[indices],blend*filtered_offsets,previous_motor,.01,lo,hi,a,b)
                    previous_motor=q[indices].copy()
                outgoing=dict(q=q,dq=np.zeros(31),kp=kp,kd=kd,tau=np.zeros(31))
                if loaded is not None and walking_state is not None and (walking_state['phase']=='carry'
                        or loaded.coordinator.transport and walking_state['phase'] in ('lower','open','done')):
                    hl=dict(live_response,heartbeat_wall=heartbeat['wall'])
                    snapshot=loaded.snapshot(state,received,numeric_mode,environment,mode_at,stamp,poses,base,
                        forces,hand_wall,hl,target,walking_state,torso_index=torso_index,
                        payload_index=payload_index,hand_indices=actual_hand_indices,
                        applied_segment=segment,control_time_s=control_time)
                    if loaded.coordinator.state=='prepared':
                        if loaded.request_pacing():
                            snapshot=loaded.refresh_output(snapshot)
                            fresh_output=snapshot.native_time-snapshot.output['sim']<.002
                            if loaded.wrist_entry_started or fresh_output:
                                outgoing=loaded.entry_command(snapshot,outgoing)
                            if fresh_output and loaded.entry_ready(snapshot):
                                loaded.request_low(snapshot)
                    else:
                        outgoing=loaded.entry_command(snapshot,outgoing)
                    if loaded.coordinator.state in ('low_pending','active'):
                        actor_snapshot=loaded.refresh_output(snapshot)
                        low_ack=(loaded.coordinator.state!='low_pending' or loaded.low_ready())
                        actor=(loaded.coordinator.advance_actor(actor_snapshot) if low_ack or numeric_mode!=3 else None)
                        if actor is not None:loaded.record_actor(actor,actor_snapshot)
                        if loaded.coordinator.state=='complete':
                            loaded_completed=True;break
                        if loaded.coordinator.state=='active':
                            pd_snapshot=loaded.prepare_pd_snapshot(snapshot)
                            if loaded.coordinator.state=='complete':
                                loaded_completed=True;break
                            frame=loaded.coordinator.compute_pd(pd_snapshot,outgoing)
                            if frame is not None:
                                send(frame.command,seq,before_publish=lambda:loaded.coordinator.check_publication(frame,loaded.fresh(pd_snapshot)))
                                loaded.coordinator.remember_published(frame)
                                published=None
                                try:
                                    published=loaded.fresh(pd_snapshot)
                                    loaded.coordinator.ack_published(frame,published)
                                finally:loaded.record_pd(frame,pd_snapshot,published)
                        else:
                            # V5 still owns legs until mode acknowledgement.
                            # During the bounded transition no stale HTD torque
                            # or position gain may survive on the owned15 axes.
                            outgoing={k:np.array(v,copy=True)for k,v in outgoing.items()}
                            outgoing['q'][:15]=loaded.coordinator.hold_q[:15]
                            for key in ('dq','kp','kd','tau'):outgoing[key][:15]=0.
                            send(outgoing,seq)
                    else:send(outgoing,seq)
                else:send(outgoing,seq)
                if walking is not None:
                    velocity=walking_state['command_xyz'] if walking_state and loaded is None else [0.,0.,0.]
                    walk_message=WalkTwist();walk_message.header.seq=seq
                    walk_message.twist.linear.x=float(velocity[0]);walk_message.twist.linear.y=float(velocity[1]);walk_message.twist.angular.z=float(velocity[2])
                    walk_publisher.publish(walk_message)
                if sample_index>=0:seen.add(sample_index)
                result['samples'].append({'wall':time.time(),'sim':float(stamp[1]),'t':age,'q':state['q'],'target':target.tolist(),'pelvis':poses[base].tolist(),'mode':mode,'archive_index':sample_index,'motion_time':motion_time,'state_receive_age_s':time.monotonic()-received})
                if live_ik:result['samples'][-1].update(nominal_target=nominal_target.tolist(),live_ik_active=live_active,
                    control_time=control_time,phase_rate=phase_rate)
                if compensate_backlash:result['samples'][-1].update(motor_target=q.tolist(),joint_offsets=offsets.tolist())
                if walking_state is not None:result['samples'][-1]['walking']=walking_state
                seq+=1;time.sleep(max(0,start+seq*.01-time.monotonic()))
        if loaded is not None:
            result.update(htd_loaded_zero=True,loaded_zero_completed=loaded_completed and loaded.coordinator.goal_hold is None,
                loaded_goal_hold_completed=loaded_completed and loaded.coordinator.goal_hold is not None and not loaded.coordinator.transport,
                loaded_transport_completed=loaded_completed and loaded.coordinator.transport,commands_sent=seq,
                scope=loaded.coordinator.contract['scope'])
            if loaded.coordinator.transport:result['walking_result']=walking_state
            if not loaded_completed:raise RuntimeError('Loaded diagnostic did not complete')
            if not result.get('htd_loaded_cleanup',{}).get('pause_acknowledged'):
                raise RuntimeError('Loaded completion requires acknowledged native pause')
            result['pass']=True
            return
        idx=[JOINTS.index(name) for name in targets]
        held=[s for s in result['samples'] if s['archive_index']>=0] if motion else [s for s in result['samples'] if s['t']>=duration]
        error=np.array([[s['q'][i]-s.get('motor_target',s['target'])[i] for i in idx] for s in held])
        if compensate_backlash:
            physical_error=np.array([[s['q'][i]+s['joint_offsets'][i]-s['target'][i] for i in idx] for s in held])
            result['output_angle_error_max_rad']=float(np.max(abs(physical_error)))
        if motion:
            result.update(archive_samples_sent=len(seen),all_archive_samples_sent=len(seen)==len(stamps))
            if walking is None and len(seen)!=len(stamps):raise RuntimeError('Replay did not finish all archived samples')
            if walking is not None:
                result['walking_result']=walking_state
                if inplace_transfer:result['inplace_result']=walking_state
                if walking_state is None or not walking_state['done']:raise RuntimeError('Walking transfer did not finish')
        result.update(pass_=bool(np.max(abs(error))<.08), rms_error_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),max_error_rad=float(np.max(abs(error))),commands_sent=seq)
        result['pass']=result.pop('pass_')
        if not result['pass']:raise RuntimeError('Upper-body hold error exceeded 0.08 rad')
    except Exception as exc:
        result['error']=str(exc);raise
    finally:
        output.write_text(json.dumps(result,indent=2))
        print('COMMAND_RESULT',output, json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','bringup','command','worker','stop'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--image',default='igris-sim:jazzy')
    p.add_argument('--native-draw',action='store_true',help='Also draw the hidden MuJoCo viewer for comparison')
    p.add_argument('--q',type=json.loads,default={'waist_yaw':.05,'l_elbow_pitch':-.15,'r_elbow_pitch':-.15})
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--motion',type=Path,help='Replay unchanged 100 Hz q archive using native simulation time')
    p.add_argument('--gains',type=Path,help='Saved replay gain TOML, required with --motion')
    p.add_argument('--compensate-backlash',action='store_true',help='Track archive output angles using the existing bridge native backlash reader')
    p.add_argument('--physics-cpu',type=int,help='Bind the physics thread to this CPU; default: fastest available core')
    p.add_argument('--live-ik',action='store_true',help='Use the fresh, validated host HL-IK arm response for measured torso compensation')
    p.add_argument('--walk-to-rolltainer',action='store_true',help='After a qualified lift, carry with WALK_LOW and release only on measured support')
    p.add_argument('--inplace-transfer',action='store_true',help='Zero WalkTwist; explicit waist arc then measured-support release')
    args=p.parse_args();out=args.out.resolve()
    if args.action=='start':start(out,args.image,args.native_draw)
    elif args.action=='bringup':bringup(out,args.physics_cpu)
    elif args.action in ('command','worker'):
        if args.action=='worker' and not Path('/.dockerenv').exists():raise RuntimeError('Worker must run in the private container')
        command(out,args.q,args.seconds,args.motion.resolve() if args.motion else None,args.gains.resolve() if args.gains else None,args.compensate_backlash,args.live_ik,args.walk_to_rolltainer,args.inplace_transfer)
    else:
        cfg=config(out)
        if cfg.get('hand_controller_pid'):
            from hands import CONTROL
            proc=Path('/proc')/str(cfg['hand_controller_pid'])/'cmdline'
            if proc.exists():
                argv=proc.read_bytes().split(b'\0')
                if str(CONTROL).encode() not in argv or str(out).encode() not in argv:raise RuntimeError('MCP process ownership changed')
                os.kill(cfg['hand_controller_pid'],signal.SIGINT)
                time.sleep(.15)
        execute(['docker','stop','--timeout','5',cfg['container']])


if __name__=='__main__':main()
