"""Fresh empty-hand state installation must never become a replay reset hook."""
import json
import os
from pathlib import Path
import resource
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DRIVER=r'''
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
int main(int argc,char**argv){
 char error[1024];auto m=mj_loadXML(argv[1],nullptr,error,1024);if(!m)return 2;
 auto d=mj_makeData(m);std::string dir=argv[2];int n=mj_sizeModel(m);
 std::vector<char>before(n),after(n);mj_saveModel(m,nullptr,before.data(),n);
 d->time=2.;mj_step(m,d);
 auto f=std::fopen((dir+"/htd_initial_state.request").c_str(),"w");std::fputs("initialize-on-empty-reset\n",f);std::fclose(f);
 mj_step(m,d);if(d->qpos[7]!=0)return 3;
 mj_resetData(m,d);mj_step(m,d);
 std::printf("FIRST %.17g %.17g %.17g\n",d->qpos[2],d->qpos[7],d->qpos[12]);
 d->qpos[7]+=.02;mj_step(m,d);std::printf("LATER %.17g\n",d->qpos[7]);
 mj_saveModel(m,nullptr,after.data(),n);if(before!=after)return 4;
 if(argc>3){mj_resetData(m,d);mj_step(m,d);}
 return 0;
}
'''


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    p=tmp_path_factory.mktemp('native-htd-initial')
    inc=ROOT/'igris/ws/install/igris_c_mujoco/include';lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    (p/'model.xml').write_text('<mujoco><option gravity="0 0 0" timestep=".001"/><default><geom contype="0" conaffinity="0"/></default><worldbody>'
        '<body name="pelvis" pos="0 0 .9"><freejoint name="base"/><geom type="sphere" size=".01" mass="1"/>'
        '<body name="arm"><joint name="hinge" axis="0 0 1"/><geom type="sphere" size=".01" mass="1"/>'
        '<body name="passive"><joint name="ball" type="ball"/><geom type="sphere" size=".01" mass="1"/></body>'
        '</body></body><body name="room_box" pos="2 0 .5"><freejoint name="room_free"/>'
        '<geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>')
    (p/'driver.cpp').write_text(DRIVER)
    subprocess.run(['g++','-O2','-std=c++17',f'-I{inc}',str(p/'driver.cpp'),f'-L{lib}',f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-o',str(p/'driver')],check=True)
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(HERE/'htd_initial_state.cpp'),'-ldl','-o',str(p/'hook.so')],check=True)
    return p


def invoke(native,tmp_path,content,*,preexisting=False,repeat=False):
    (tmp_path/'htd_initial_state.txt').write_text(content)
    if preexisting:(tmp_path/'htd_initial_state.request').write_text('initialize-on-empty-reset\n')
    env=dict(os.environ,LD_PRELOAD=str(native/'hook.so'),IGRIS_HTD_INITIAL_STATE_DIR=str(tmp_path))
    def disable_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    return subprocess.run([str(native/'driver'),str(native/'model.xml'),str(tmp_path)]+(['repeat']if repeat else []),
        env=env,text=True,capture_output=True,preexec_fn=disable_core)


STATE='3\nbase 7 .1 -.2 .95 1 0 0 0\nhinge 1 .3\nball 4 1 0 0 0\n'


def test_only_robot_state_on_one_observed_reset_and_model_unchanged(native,tmp_path):
    r=invoke(native,tmp_path,STATE);assert r.returncode==0,r.stderr
    first=next(l.split()[1:]for l in r.stdout.splitlines()if l.startswith('FIRST'))
    assert list(map(float,first))==pytest.approx([.95,.3,2.])
    assert 'LATER 0.320' in r.stdout or 'LATER 0.32' in r.stdout
    ack=json.loads((tmp_path/'htd_initial_state_applied.json').read_text())
    assert ack['robot_joints']==3 and ack['room_qpos_unchanged'] and ack['apply_count']==1


@pytest.mark.parametrize('case',['preexisting','repeat','room','missing','nonfinite','quaternion'])
def test_reject_unsafe_scope_or_reuse(native,tmp_path,case):
    text=STATE
    if case=='room':text+='room_free 7 5 0 0 1 0 0 0\n';text=text.replace('3\n','4\n',1)
    if case=='missing':text=text.replace('3\n','2\n',1).replace('ball 4 1 0 0 0\n','')
    if case=='nonfinite':text=text.replace('hinge 1 .3','hinge 1 nan')
    if case=='quaternion':text=text.replace('ball 4 1 0 0 0','ball 4 2 0 0 0')
    r=invoke(native,tmp_path,text,preexisting=case=='preexisting',repeat=case=='repeat')
    assert r.returncode!=0
