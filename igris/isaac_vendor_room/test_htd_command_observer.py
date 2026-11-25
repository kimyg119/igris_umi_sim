"""Real compiled ABI fixture: catches changed arguments, duplicate calls and logging loss."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def module():
    path = HERE/'htd_command_observer.py'
    assert path.is_file(), 'Logging-only observer build/parser is not implemented'
    spec = importlib.util.spec_from_file_location('htd_command_observer', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


DECL = r'''
#include "igris_c_lib/kinematics/JointManager.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
using JM = robros::controller::igris_c::JointManager;
using RC = robros::shm::command::RobotCommand;
using MS = robros::shm::state::motorStatusData_st;
using CM = std::array<ParallelLinkCommandMode, 5>;
using IM = std::array<ParallelLinkIKSolverMode, 5>;
using SM = std::array<ParallelLinkSafetyMode, 5>;
using SS = std::array<ParallelLinkSoftSafeSetMode, 5>;
using BB = std::array<bool, 5>;
using Fn = int (*)(JM*, const MS&, const RC&, RC&, const CM&, const IM&,
                  const SM&, const SS&, const BB&, const BB&);
constexpr char symbol[] = "_ZN6robros10controller7igris_c12JointManager18jointCommand2MotorERKNS_3shm5state18motorStatusData_stERKNS3_7command12RobotCommandERS9_RKSt5arrayI23ParallelLinkCommandModeLm5EERKSD_I24ParallelLinkIKSolverModeLm5EERKSD_I22ParallelLinkSafetyModeLm5EERKSD_I27ParallelLinkSoftSafeSetModeLm5EERKSD_IbLm5EESW_";
'''

PROVIDER = DECL + r'''
extern "C" __attribute__((visibility("default"))) int provider(JM*, const MS&, const RC&, RC&, const CM&, const IM&,
                        const SM&, const SS&, const BB&, const BB&) asm("_ZN6robros10controller7igris_c12JointManager18jointCommand2MotorERKNS_3shm5state18motorStatusData_stERKNS3_7command12RobotCommandERS9_RKSt5arrayI23ParallelLinkCommandModeLm5EERKSD_I24ParallelLinkIKSolverModeLm5EERKSD_I22ParallelLinkSafetyModeLm5EERKSD_I27ParallelLinkSoftSafeSetModeLm5EERKSD_IbLm5EESW_");
extern "C" { __attribute__((visibility("default"))) int calls = 0; }
int provider(JM* self, const MS& ms, const RC& from, RC& to, const CM& cm,
             const IM& im, const SM& sm, const SS& ss, const BB& gain, const BB& ff) {
  // An opaque sentinel tests pointer forwarding; no JointManager is constructed or accessed.
  if (reinterpret_cast<uintptr_t>(self) != 0x12340) { std::fprintf(stderr,"wrong self %p\n",self); std::abort(); }
  ++calls;
  RC copy = from;  // Deliberately support aliased from/to.
  to = copy;
  for (int i=0; i<31; ++i) {
    to.q[i] += ms.current_position[i]; to.qDot[i] -= ms.current_velocity[i];
    to.torque[i] += 10; to.kp[i] += 20; to.kd[i] += 30;
    to.controlMode[i] = static_cast<robros::shm::command::control_mode>((i+2)%8);
  }
  for (int i=0; i<5; ++i) {
    if (int(cm[i]) != i%2 || int(im[i]) != (i+1)%2 || int(sm[i]) != (i/2)%2 ||
        int(ss[i]) != ((i+2)/2)%2 || gain[i] != bool(i%2) || ff[i] != bool((i+1)%2)) {
      std::fprintf(stderr,"wrong options %d: %d %d %d %d %d %d\n",i,int(cm[i]),int(im[i]),int(sm[i]),int(ss[i]),int(gain[i]),int(ff[i])); std::abort(); }
    to.kinematicModes[i] = static_cast<robros::shm::command::KinematicMode>((i+1)%2);
  }
  to.header.seq += 100; to.header.frame_id[0] = 'Z';
  errno = ERANGE;
  return -37;
}
'''

DRIVER = DECL + r'''
int main(int argc, char** argv) {
  void* handle = dlopen(argv[1], RTLD_NOW|RTLD_GLOBAL);
  if (!handle) return 90;
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, symbol));
  auto calls = reinterpret_cast<int*>(dlsym(handle, "calls"));
  if (!fn || !calls) return 91;
  MS ms;
  RC input, output;
  std::memset(&input.header, 0, sizeof(input.header));
  std::memset(&ms.header, 0, sizeof(ms.header));
  input.header.seq=23; input.header.sec=42; input.header.nanosec=314;
  std::strcpy(input.header.frame_id,"serial-input");
  ms.header.seq=99; ms.header.sec=55; ms.header.nanosec=271;
  std::strcpy(ms.header.frame_id,"motor-status");
  CM cm; IM im; SM sm; SS ss; BB gain, ff;
  for (int i=0;i<5;++i) {
    cm[i]=static_cast<ParallelLinkCommandMode>(i%2);
    im[i]=static_cast<ParallelLinkIKSolverMode>((i+1)%2);
    sm[i]=static_cast<ParallelLinkSafetyMode>((i/2)%2);
    ss[i]=static_cast<ParallelLinkSoftSafeSetMode>(((i+2)/2)%2);
    gain[i]=i%2; ff[i]=(i+1)%2;
    input.kinematicModes[i]=static_cast<robros::shm::command::KinematicMode>(i%2);
  }
  for(int i=0;i<31;++i) {
    input.q[i]=i+.25; input.qDot[i]=i+.5; input.torque[i]=i+.75;
    input.kp[i]=i+100.; input.kd[i]=i+200.;
    input.controlMode[i]=static_cast<robros::shm::command::control_mode>(i%8);
    ms.motor_id[i]=1000+i; std::snprintf(ms.joint_name[i],30,"joint-%d",i);
    std::snprintf(ms.motor_name[i],30,"motor-%d",i);
    ms.target_position[i]=i+10.5; ms.target_torque[i]=i+20.5;
    ms.current_position[i]=i+30.5; ms.current_velocity[i]=i+40.5;
    ms.current_acceleration[i]=i+50.5; ms.current_torque[i]=i+60.5;
    ms.current_current[i]=i+70.5; ms.current_temp[i]=i+80.5;
    ms.error_state[i]=2000+i; ms.packet_rate[i]=3000+i;
  }
  const RC original=input;
  FILE* result=std::fopen(argv[2],"wb");
  int n=std::atoi(argv[3]), arm_after=std::atoi(argv[4]);
  for(int k=0;k<n;++k) {
    if(k==arm_after && std::getenv("IGRIS_HTD_COMMAND_OBSERVER_ARM")) {
      int fd=open(std::getenv("IGRIS_HTD_COMMAND_OBSERVER_ARM"),O_CREAT|O_WRONLY,0600);
      if(fd>=0) close(fd);
    }
    input=original;
    bool alias=(k%2)==1;
    RC& target=alias?input:output;
    errno=EDOM;
    int rc=fn(reinterpret_cast<JM*>(0x12340),ms,input,target,cm,im,sm,ss,gain,ff);
    if(rc!=-37 || errno!=ERANGE || *calls!=k+1) return 92;
    if(!alias && std::memcmp(&input,&original,sizeof(input))) return 93;
    std::fwrite(target.q.data(),sizeof(double),31,result);
    std::fwrite(target.qDot.data(),sizeof(double),31,result);
    std::fwrite(target.torque.data(),sizeof(double),31,result);
    std::fwrite(target.kp.data(),sizeof(double),31,result);
    std::fwrite(target.kd.data(),sizeof(double),31,result);
    std::fwrite(target.controlMode,sizeof(target.controlMode),1,result);
    std::fwrite(target.kinematicModes,sizeof(target.kinematicModes),1,result);
    std::fwrite(&target.header,sizeof(target.header),1,result);
  }
  std::fclose(result);
  return 0;
}
'''


@pytest.fixture(scope='module')
def compiled():
    mod = module()
    # Actual headers depend on the supplied vendor image; compilation only, no physics.
    out = Path(tempfile.mkdtemp(prefix='command_observer_test_', dir=ROOT/'outputs'))
    try:
        evidence = mod.build(out, max_records=3)
        (out/'provider.cpp').write_text(PROVIDER)
        (out/'driver.cpp').write_text(DRIVER)
        mod.compile_cpp(out/'provider.cpp', out/'provider.so', shared=True)
        mod.compile_cpp(out/'driver.cpp', out/'driver', shared=False)
        yield mod, out, evidence
    finally:
        shutil.rmtree(out)


def execute(compiled, tmp_path, *, armed=True, max_records=3, wrong_exe=False, n=5):
    mod, built, evidence = compiled
    # The fixture binary and .so use only standard C++/dl; execute host-side, without vendor runtime.
    local = tmp_path/'trace.bin'
    arm = tmp_path/'arm'
    env = os.environ.copy()
    env.update(evidence['environment'])
    # Vendor provider is a startup dependency of SIM. Match that ELF lookup scope.
    env['LD_PRELOAD'] += ':'+str(built/'provider.so')
    env.update(IGRIS_HTD_COMMAND_OBSERVER_ARM=str(arm),
               IGRIS_HTD_COMMAND_OBSERVER_TRACE=str(local),
               IGRIS_HTD_COMMAND_OBSERVER_EXE=str(built/'driver') if not wrong_exe else '/wrong/sim',
               IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS=str(max_records))
    command = [str(built/'driver'), str(built/'provider.so'), str(tmp_path/'observed'), str(n), '1' if armed else '-1']
    subprocess.run(command, env=env, check=True)
    clean = env.copy()
    clean['LD_PRELOAD']=str(built/'provider.so')
    command[2]=str(tmp_path/'baseline')
    subprocess.run(command, env=clean, check=True)
    assert (tmp_path/'observed').read_bytes() == (tmp_path/'baseline').read_bytes()
    return mod, local


def test_unarmed_has_no_trace_and_preserves_forwarded_result(compiled, tmp_path):
    _, path = execute(compiled, tmp_path, armed=False)
    assert not path.exists()


def test_wrong_executable_cannot_arm(compiled, tmp_path):
    _, path = execute(compiled, tmp_path, wrong_exe=True)
    assert not path.exists()


def test_armed_alias_and_error_return_capture_all_fields_with_bounded_records(compiled, tmp_path):
    mod, path = execute(compiled, tmp_path)
    trace = mod.read_trace(path)
    records = trace['records']
    assert len(records)==3 and trace['capacity']==3
    assert trace['dropped_capacity']==1
    assert path.stat().st_size == trace['expected_bytes']
    np.testing.assert_array_equal(records['sequence'], [1,2,3])
    np.testing.assert_array_equal(records['return_code'], [-37,-37,-37])
    assert np.all(records['exit_monotonic_ns'] >= records['entry_monotonic_ns'])
    for row in records:
        before, after, motor = row['before'], row['after'], row['motor']
        assert before['header']['seq']==23 and after['header']['seq']==123
        assert before['header']['frame_id']==b'serial-input'
        assert after['header']['frame_id']==b'Zerial-input'
        assert motor['header']['seq']==99 and motor['header']['frame_id']==b'motor-status'
        np.testing.assert_array_equal(before['q'], np.arange(31)+.25)
        np.testing.assert_array_equal(after['q'], 2*np.arange(31)+30.75)
        np.testing.assert_array_equal(before['dq'], np.arange(31)+.5)
        np.testing.assert_array_equal(after['dq'], np.full(31,-40.))
        for key, before_offset, after_offset in [('tau',.75,10.75),('kp',100.,120.),('kd',200.,230.)]:
            np.testing.assert_array_equal(before[key], np.arange(31)+before_offset)
            np.testing.assert_array_equal(after[key], np.arange(31)+after_offset)
        np.testing.assert_array_equal(before['control_mode'], np.arange(31)%8)
        np.testing.assert_array_equal(after['control_mode'], (np.arange(31)+2)%8)
        np.testing.assert_array_equal(before['kinematic_mode'], [0,1,0,1,0])
        np.testing.assert_array_equal(after['kinematic_mode'], [1,0,1,0,1])
        np.testing.assert_array_equal(row['options'], [[0,1,0,1,0],[1,0,1,0,1],[0,0,1,1,0],
                                                    [1,1,0,0,1],[0,1,0,1,0],[1,0,1,0,1]])
        for key, offset in [('target_position',10.5),('target_torque',20.5),('current_position',30.5),
                            ('current_velocity',40.5),('current_acceleration',50.5),('current_torque',60.5),
                            ('current_current',70.5),('current_temp',80.5),('motor_id',1000),
                            ('error_state',2000),('packet_rate',3000)]:
            np.testing.assert_array_equal(motor[key], np.arange(31)+offset)
        assert motor['joint_name'][30]==b'joint-30' and motor['motor_name'][30]==b'motor-30'


def test_existing_trace_is_never_overwritten(compiled, tmp_path):
    path=tmp_path/'trace.bin'
    path.write_bytes(b'original-evidence')
    execute(compiled,tmp_path)
    assert path.read_bytes()==b'original-evidence'


def test_invalid_record_bound_disables_logging_only(compiled, tmp_path):
    _, path=execute(compiled,tmp_path,max_records=8193)
    assert not path.exists()


def test_parser_rejects_corrupt_header_and_incomplete_records(compiled,tmp_path):
    mod,path=execute(compiled,tmp_path)
    raw=path.read_bytes()
    path.write_bytes(raw[:10])
    with pytest.raises(ValueError,match='header'):
        mod.read_trace(path)
    path.write_bytes(b'badmagic'+raw[8:])
    with pytest.raises(ValueError,match='format'):
        mod.read_trace(path)


def test_parser_excludes_uncommitted_final_record(compiled,tmp_path):
    mod,path=execute(compiled,tmp_path)
    trace=mod.read_trace(path)
    raw=bytearray(path.read_bytes())
    # Simulates interrupted publication of the final record, not a shortened valid file.
    at=trace['header_bytes']+2*trace['record_bytes']
    raw[at:at+8]=b'\x00'*8
    path.write_bytes(raw)
    parsed=mod.read_trace(path)
    assert len(parsed['records'])==2 and parsed['uncommitted_records']==1
