"""One-step native3.2.7 fixture; verifies solved-force grouping without a full run."""
import json
import os
from pathlib import Path
import resource
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DRIVER=r'''
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>
int main(int argc,char**argv){
 if(std::strcmp(mj_versionString(),"3.2.7"))return 2;
 char error[1024];auto m=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!m){std::fprintf(stderr,"%s",error);return 3;}
 // Model compilation can invoke mj_step on scratch data. Ensure the 30 Hz
 // observer samples the real one-step fixture after that compiler activity.
 auto d=mj_makeData(m);usleep(40000);mj_step(m,d);
 for(int i=0;i<m->nq;i++)std::printf("%.17g ",d->qpos[i]);
 for(int i=0;i<m->nv;i++)std::printf("%.17g ",d->qvel[i]);std::puts("");
 mj_deleteData(d);mj_deleteModel(m);
}
'''


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    out=tmp_path_factory.mktemp('observer-support')
    (out/'model.xml').write_text('''<mujoco><option gravity="0 0 0" timestep=".001"/><worldbody>
      <body name="room_source" pos="-.2 0 .05"><freejoint/><geom name="room_geom_source" type="box" size=".06 .07 .05" mass="1000"/></body>
      <body name="room_destination" pos="0 0 .05"><freejoint/><geom name="room_geom_destination" type="box" size=".06 .07 .05" mass="1000"/></body>
      <body name="r_forearm" pos=".2 0 .05"><geom name="forearm_geom" type="box" size=".06 .07 .05"/></body>
      <body name="room_cart" pos="2 0 .05"><geom name="cart_geom" type="box" size=".1 .1 .05"/></body>
      <body name="room_payload" pos="0 0 .14"><freejoint/><geom name="room_geom_payload" type="box" size=".31 .08 .05" mass="1"/></body>
    </worldbody></mujoco>''')
    (out/'driver.cpp').write_text(DRIVER)
    inc=ROOT/'igris/ws/install/igris_c_mujoco/include';lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    subprocess.run(['g++','-O2','-std=c++17',f'-I{inc}',str(out/'driver.cpp'),f'-L{lib}',f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-o',str(out/'driver')],check=True)
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',str(HERE/'observer.cpp'),'-ldl','-o',str(out/'observer.so')],check=True)
    return out


def run(native,out,source='room_source\n',destination='room_destination\n',observer=True):
    env=os.environ.copy()
    for name in ('LD_PRELOAD','IGRIS_ROOM_OUTPUT','IGRIS_ROOM_PAYLOAD','IGRIS_ROOM_SOURCE_SUPPORT_BODIES_FILE','IGRIS_ROOM_DESTINATION_SUPPORT_BODIES_FILE'):env.pop(name,None)
    if observer:
        env.update(LD_PRELOAD=str(native/'observer.so'),IGRIS_ROOM_OUTPUT=str(out),IGRIS_ROOM_PAYLOAD='room_payload')
        for name,text in [('SOURCE',source),('DESTINATION',destination)]:
            if text is not None:
                path=out/(name.lower()+'.txt');path.write_text(text)
                env['IGRIS_ROOM_'+name+'_SUPPORT_BODIES_FILE']=str(path)
    def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    return subprocess.run([str(native/'driver'),str(native/'model.xml')],env=env,capture_output=True,text=True,preexec_fn=no_core)


def test_actual_solved_source_destination_force_excludes_forearm_and_preserves_native_state(native,tmp_path):
    result=run(native,tmp_path);assert result.returncode==0,result.stderr
    grasp=json.loads((tmp_path/'grasp_force.json').read_text())
    assert grasp['support_force_groups_configured']
    source=grasp['source_support_force_world_z_N'];dest=grasp['destination_support_force_world_z_N']
    assert source>0 and dest>0
    assert grasp['payload_support_force_world_z_N']>source+dest+1.
    events=[json.loads(s)for s in(tmp_path/'payload_contacts.jsonl').read_text().splitlines()]
    expected={name:sum(e['force_on_payload_world_N'][2]for e in events if name in e['geoms'])for name in('room_geom_source','room_geom_destination')}
    assert source==pytest.approx(expected['room_geom_source'],rel=1e-7)
    assert dest==pytest.approx(expected['room_geom_destination'],rel=1e-7)
    assert json.loads((tmp_path/'support_force_groups.json').read_text())['source_bodies']==['room_source']
    without=run(native,tmp_path,observer=False);assert without.returncode==0
    np.testing.assert_array_equal(np.fromstring(result.stdout,sep=' '),np.fromstring(without.stdout,sep=' '))


def test_legacy_without_support_selectors_keeps_total_and_marks_groups_unconfigured(native,tmp_path):
    result=run(native,tmp_path,None,None);assert result.returncode==0,result.stderr
    grasp=json.loads((tmp_path/'grasp_force.json').read_text())
    assert not grasp['support_force_groups_configured']
    assert grasp['source_support_force_world_z_N']==0 and grasp['destination_support_force_world_z_N']==0
    assert grasp['payload_support_force_world_z_N']>0


@pytest.mark.parametrize('source,destination',[
    ('','room_destination'),('room_missing','room_destination'),('room_cart','room_destination'),
    ('r_forearm','room_destination'),('room_source\nroom_source','room_destination'),
    ('room_source','room_source'),('room_payload','room_destination'),('room_source',None)])
def test_invalid_support_selectors_rejected_before_output(native,tmp_path,source,destination):
    result=run(native,tmp_path,source,destination)
    assert result.returncode!=0 and 'support force selector' in result.stderr
    assert not (tmp_path/'grasp_force.json').exists()
