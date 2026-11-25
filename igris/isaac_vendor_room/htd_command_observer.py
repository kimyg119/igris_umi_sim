"""Build and parse a bounded, opt-in SIM controller command-mapping observer.

The trace records effective commands at JointManager's public mapping boundary,
after SlowStart/Safety preprocessing. It does not expose SlowStart internal state
or final post-mapping motor safety clamps. No runtime launch is performed here.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'igris-c-vanilla:2.1.0post1'
INSTALL = ROOT/'igris/ws/install'
SIM = INSTALL/'igris_c_controller/lib/igris_c_controller/sim'
LIB = INSTALL/'igris_c_lib/lib/libigris_c_lib.so'
SYMBOL = ('_ZN6robros10controller7igris_c12JointManager18jointCommand2MotorERKNS_3shm5state18motorStatusData_st'
          'ERKNS3_7command12RobotCommandERS9_RKSt5arrayI23ParallelLinkCommandModeLm5EERKSD_I24ParallelLinkIKSolverModeLm5E'
          'ERKSD_I22ParallelLinkSafetyModeLm5EERKSD_I27ParallelLinkSoftSafeSetModeLm5EERKSD_IbLm5EESW_')

HEADER = np.dtype([('seq','<u4'),('sec','<u4'),('nanosec','<u4'),('frame_id','S256')])
COMMAND = np.dtype([('header',HEADER),('kinematic_mode','<u4',(5,)),('control_mode','<u4',(31,))]
                   +[(key,'<f8',(31,)) for key in ('q','dq','tau','kp','kd')])
MOTOR = np.dtype([('header',HEADER),('motor_id','<u4',(31,)),('joint_name','S30',(31,)),('motor_name','S30',(31,))]
                 +[(key,'<f8',(31,)) for key in ('target_position','target_torque','current_position','current_velocity',
                                               'current_acceleration','current_torque','current_current','current_temp')]
                 +[('error_state','<u4',(31,)),('packet_rate','<u8',(31,))])
RECORD = np.dtype([('sequence','<u8'),('entry_monotonic_ns','<u8'),('exit_monotonic_ns','<u8'),
                   ('return_code','<i4'),('reserved','<u4'),('options','<i4',(6,5)),
                   ('before',COMMAND),('after',COMMAND),('motor',MOTOR)])
FILE_HEADER = np.dtype([('magic','S8'),('version','<u4'),('header_bytes','<u4'),('record_bytes','<u4'),
                       ('capacity','<u4'),('reserved_slots','<u8'),('dropped_capacity','<u8'),
                       ('dropped_busy','<u8'),('armed_monotonic_ns','<u8'),('pid','<u8')])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_cpp(source, destination, *, shared=True):
    """Compile against installed public headers in the existing vendor build image only."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    source.relative_to(ROOT)
    destination.relative_to(ROOT/'outputs')
    def mounted(path):
        return '/project/'+str(path.relative_to(ROOT))
    includes = ['-I/usr/include/eigen3','-I/opt/ros/jazzy/include/tf2']
    includes += ['-I/project/igris/ws/install/'+name+'/include'
                 for name in ('igris_c_lib','igris_c','igris_c_shm')]
    command = ['docker','run','--rm','--network','none','--cpuset-cpus','4,7',
               '--user',f'{os.getuid()}:{os.getgid()}',
               '--mount',f'type=bind,src={ROOT},dst=/project,readonly',
               '--mount',f'type=bind,src={destination.parent},dst={mounted(destination.parent)}',
               '--entrypoint','g++',IMAGE,'-O2','-std=c++17','-DFMT_HEADER_ONLY',*includes,str(mounted(source)),
               *(['-shared','-fPIC','-fvisibility=hidden'] if shared else []),
               '-ldl','-pthread','-o',mounted(destination)]
    result=subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Command observer compilation failed:\n'+result.stderr)
    return command


def build(out, *, max_records=4096):
    """Create a trial-local library/manifest; caller alone wires SIM preload and arms it."""
    if type(max_records) is not int or not 1 <= max_records <= 8192:
        raise ValueError('max_records must be an integer in 1..8192')
    out=Path(out).resolve()
    out.relative_to(ROOT/'outputs')
    out.mkdir(parents=True,exist_ok=True)
    library=out/'htd_command_observer.so'
    arm=out/'htd_command_observer.arm'
    trace=out/'htd_command_observer.bin'
    manifest=out/'htd_command_observer.json'
    if any(path.exists() for path in (library,arm,trace,manifest)):
        raise FileExistsError('Command observer requires fresh trial-local paths')
    source=Path(__file__).with_suffix('.cpp')
    command=compile_cpp(source,library)
    exported=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
    if SYMBOL not in exported:
        raise RuntimeError('Observer is missing the exact installed mapping symbol')
    environment=dict(LD_PRELOAD=str(library),IGRIS_HTD_COMMAND_OBSERVER_ARM=str(arm),
                     IGRIS_HTD_COMMAND_OBSERVER_TRACE=str(trace),IGRIS_HTD_COMMAND_OBSERVER_EXE=str(SIM.resolve()),
                     IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS=str(max_records))
    headers=[INSTALL/'igris_c_lib/include/igris_c_lib/kinematics/JointManager.h',
             INSTALL/'igris_c_shm/include/igris_c_shm/shm/command.h',
             INSTALL/'igris_c_shm/include/igris_c_shm/shm/state.h',
             INSTALL/'igris_c_shm/include/igris_c_shm/shm/common.h']
    result=dict(library=str(library),arm_path=str(arm),trace_path=str(trace),environment=environment,
                sha256=sha256(library),source_sha256=sha256(source),python_sha256=sha256(__file__),
                input_sha256={str(p):sha256(p) for p in [SIM,LIB,*headers]},
                image=IMAGE,image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',IMAGE],text=True).strip(),
                build_command=command,
                contract=dict(schema='htd_command_observer/v1',symbol=SYMBOL,logging_only=True,
                              controller_executable_only=True,arm_once=True,max_records=max_records,
                              record_bytes=RECORD.itemsize,header_bytes=FILE_HEADER.itemsize,
                              maximum_bytes=FILE_HEADER.itemsize+max_records*RECORD.itemsize,
                              fields='All public input/output RobotCommand and motorStatusData fields; all six mapping option arrays',
                              boundary='Full JointManager::jointCommand2Motor RobotCommand overload, after upstream SlowStart/Safety',
                              not_observed='SlowStart internal active flag; final downstream motor safety clamps; physics-applied effort',
                              persistence='MAP_SHARED committed records; no per-frame file syscall/flush; no power-loss guarantee'))
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    return result


def read_trace(path):
    """Read committed records from a stopped or live bounded file without object ABI guesses."""
    raw=Path(path).read_bytes()
    if len(raw)<FILE_HEADER.itemsize:
        raise ValueError('Incomplete command observer header')
    header=np.frombuffer(raw,dtype=FILE_HEADER,count=1)[0]
    capacity=int(header['capacity'])
    if (header['magic']!=b'HTDCMD01' or header['version']!=1 or
        header['header_bytes']!=FILE_HEADER.itemsize or header['record_bytes']!=RECORD.itemsize or
        not 1<=capacity<=8192):
        raise ValueError('Unsupported command observer format')
    expected=FILE_HEADER.itemsize+capacity*RECORD.itemsize
    if len(raw)!=expected:
        raise ValueError('Incomplete command observer record allocation')
    all_records=np.frombuffer(raw,dtype=RECORD,count=capacity,offset=FILE_HEADER.itemsize)
    complete=all_records['sequence']==np.arange(1,capacity+1,dtype=np.uint64)
    nonzero=all_records['sequence']!=0
    if np.any(nonzero & ~complete):
        raise ValueError('Corrupt command observer sequence')
    reserved=min(int(header['reserved_slots']),capacity)
    return dict(records=all_records[complete].copy(),capacity=capacity,record_bytes=RECORD.itemsize,
                header_bytes=FILE_HEADER.itemsize,expected_bytes=expected,
                dropped_capacity=int(header['dropped_capacity']),dropped_busy=int(header['dropped_busy']),
                uncommitted_records=reserved-int(complete.sum()),armed_monotonic_ns=int(header['armed_monotonic_ns']),
                pid=int(header['pid']))
