"""Exercise the staging hook against native MuJoCo 3.2.7, without Docker."""
import json
import os
from pathlib import Path
import resource
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ['room_Boxes__Pallet_C1_L1', 'room_Boxes__Pallet_C1_L2',
          'room_Boxes__Small_Center_L1', 'room_Boxes__Small_Center_L2',
          'room_Boxes__Small_RightFront']

DRIVER = r'''
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
void output(const mjModel* m, const mjData* d, int phase) {
 for(int b=1;b<m->nbody;b++) {
  const char* n=mj_id2name(m,mjOBJ_BODY,b);if(!n)continue;
  std::printf("%d %s",phase,n);
  if(m->body_jntnum[b]==1) {
   int j=m->body_jntadr[b],q=m->jnt_qposadr[j],v=m->jnt_dofadr[j];
   for(int i=0;i<7;i++)std::printf(" %.17g",d->qpos[q+i]);
   for(int i=0;i<6;i++)std::printf(" %.17g",d->qvel[v+i]);
  } else {
   for(int i=0;i<3;i++)std::printf(" %.17g",d->xpos[3*b+i]);
  }
  std::printf("\n");
 }
}
void request(const std::string& dir,const char* value) {
 auto f=std::fopen((dir+"/room_layout.request").c_str(),"w");
 if(!f)std::abort();std::fputs(value,f);std::fclose(f);
}
int main(int argc,char** argv) {
 if(std::strcmp(mj_versionString(),"3.2.7"))return 2;
 char error[1024]{};auto m=mj_loadXML(argv[1],nullptr,error,1024);
 if(!m){std::fprintf(stderr,"%s",error);return 3;}
 auto d=mj_makeData(m);std::string dir=argv[2];
 mj_step(m,d);output(m,d,1);
 if(argc>3)return 0;
 // Selection is cached at initialization, even if its file changes later.
 if(auto file=std::getenv("IGRIS_ROOM_STAGE_BODIES_FILE")) {
  auto f=std::fopen(file,"w");if(!f)return 4;
  std::fputs("room_nonexistent\n",f);std::fclose(f);
 }
 for(int i=0;i<m->nv;i++)d->qvel[i]=0;
 for(int b=1;b<m->nbody;b++) {
  const char* n=mj_id2name(m,mjOBJ_BODY,b);
  if(n && !std::strncmp(n,"room_",5) && m->body_jntnum[b]==1)
   d->qvel[m->jnt_dofadr[m->body_jntadr[b]]]=1.;
 }
 request(dir,"0.2 -0.3 1.5707963267948966\n");
 mj_step(m,d);output(m,d,2);
 std::rename((dir+"/room_layout_applied.json").c_str(),(dir+"/first_applied.json").c_str());
 // Perturb state after ready; no subsequent step may stage it again.
 for(const char* n : {"room_Boxes__Small_RightFront","room_destination"}) {
  int b=mj_name2id(m,mjOBJ_BODY,n);d->qpos[m->jnt_qposadr[m->body_jntadr[b]]]+=.07;
 }
 request(dir,"0.4 0.5 0\n");
 for(int i=0;i<3;i++)mj_step(m,d);
 output(m,d,3);
 // A reset re-applies the currently published request once.
 mj_resetData(m,d);mj_step(m,d);output(m,d,4);
 // Reset immediately after that single step, now without a ready request.
 unlink((dir+"/room_layout.request").c_str());
 mj_resetData(m,d);mj_step(m,d);output(m,d,5);
 mj_deleteData(d);mj_deleteModel(m);
}
'''


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-room-stage')
    inc = ROOT / 'igris/ws/install/igris_c_mujoco/include'
    lib = ROOT / 'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    bodies = ''.join(f'<body name="{name}" pos="{1+i*.2} 0 .8"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>' for i, name in enumerate(SOURCE))
    (directory / 'model.xml').write_text('<mujoco><option gravity="0 0 0" timestep=".001"/><worldbody>'
        '<body name="robot" pos="0 0 .9"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>' + bodies +
        '<body name="room_destination" pos="0 2 .8"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>'
        '<body name="room_extra" pos="-2 -1 .3"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>'
        '<body name="room_cart" pos="0 2 0"><geom type="box" size=".4 .4 .1"/></body>'
        '</worldbody></mujoco>')
    (directory / 'driver.cpp').write_text(DRIVER)
    subprocess.run(['g++', '-O2', '-std=c++17', f'-I{inc}', str(directory/'driver.cpp'),
                    f'-L{lib}', f'-Wl,-rpath,{lib}', '-l:libmujoco.so.3.2.7', '-o', str(directory/'driver')], check=True)
    subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', f'-I{inc}',
                    str(HERE/'room_setup.cpp'), '-ldl', '-o', str(directory/'room_setup.so')], check=True)
    return directory


def run_native(native, directory, selection=None, short=False, scope=None):
    env = os.environ.copy()
    env.update(LD_PRELOAD=str(native/'room_setup.so'), IGRIS_ROOM_STAGE=str(directory))
    env.pop('IGRIS_ROOM_STAGE_BODIES_FILE', None)
    env.pop('IGRIS_ROOM_STAGE_SCOPE', None)
    if scope is not None:
        env['IGRIS_ROOM_STAGE_SCOPE'] = scope
    if selection is not None:
        path = directory/'selected_bodies.txt'
        path.write_text(selection)
        env['IGRIS_ROOM_STAGE_BODIES_FILE'] = str(path)
    command = [str(native/'driver'), str(native/'model.xml'), str(directory)]
    if short:
        command.append('short')
    def no_core_dump():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return subprocess.run(command, env=env, text=True, capture_output=True, preexec_fn=no_core_dump)


@pytest.mark.parametrize('selective', [True, False])
def test_native_staging_alignment_once_and_resets(native, tmp_path, selective):
    result = run_native(native, tmp_path, '\n'.join(SOURCE)+'\n' if selective else None)
    assert result.returncode == 0, result.stderr
    records = {}
    for line in result.stdout.splitlines():
        phase, body, *values = line.split()
        records[int(phase), body] = np.array(values, float)
    for phase in range(1, 6):
        np.testing.assert_allclose(records[phase, 'room_cart'], [0., 2., 0.], atol=1e-12)
        np.testing.assert_allclose(records[phase, 'robot'][:3], [0., 0., .9], atol=1e-12)
    for i, body in enumerate(SOURCE):
        np.testing.assert_allclose(records[1, body][:3], [11+i*.2, 0., .8], atol=1e-12)
        np.testing.assert_allclose(records[2, body][:3], [.2, .7+i*.2, .8], atol=1e-12)
        np.testing.assert_allclose(records[2, body][3:7], [2**-.5, 0., 0., 2**-.5], atol=1e-12)
        np.testing.assert_array_equal(records[2, body][7:], np.zeros(6))
        np.testing.assert_allclose(records[4, body][:3], [1.4+i*.2, .5, .8], atol=1e-12)
        np.testing.assert_allclose(records[5, body][:3], [11+i*.2, 0., .8], atol=1e-12)
    target_ready = [0., 2., .8] if selective else [-1.8, -.3, .8]
    extra_ready = [-2., -1., .3] if selective else [1.2, -2.3, .3]
    np.testing.assert_allclose(records[2, 'room_destination'][:3], target_ready, atol=1e-12)
    np.testing.assert_allclose(records[2, 'room_extra'][:3], extra_ready, atol=1e-12)
    np.testing.assert_array_equal(records[2, 'room_destination'][7:], np.zeros(6))
    for body in (SOURCE[-1], 'room_destination'):
        np.testing.assert_allclose(records[3, body][:3], records[2, body][:3]+[.07, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(records[4, 'room_destination'][:3], [0., 2., .8] if selective else [.4, 2.5, .8], atol=1e-12)
    np.testing.assert_allclose(records[5, 'room_destination'][:3], [10., 2., .8], atol=1e-12)
    applied = json.loads((tmp_path/'first_applied.json').read_text())
    if selective:
        assert applied['scope'] == 'source_only'
        assert applied['transformed_body_count'] == 5
        assert set(applied['transformed_bodies']) == set(SOURCE)
        assert applied['fixed_room_free_bodies'] == 2
        assert not applied['transforms_destination']


@pytest.mark.parametrize('selection', ['', '\n \t\n', 'room_missing\n', 'room_cart\n',
                                        'robot\n', SOURCE[0]+'\n'+SOURCE[0]+'\n'])
def test_native_rejects_invalid_selector(native, tmp_path, selection):
    result = run_native(native, tmp_path, selection, short=True)
    assert result.returncode != 0
    assert not result.stdout
    assert 'room staging selector' in result.stderr


def test_native_explicit_near_scope_transforms_destination_and_reports_names(native,tmp_path):
    names=SOURCE+['room_destination']
    result=run_native(native,tmp_path,'\n'.join(names)+'\n',scope='source_and_destination')
    assert result.returncode==0,result.stderr
    applied=json.loads((tmp_path/'first_applied.json').read_text())
    assert applied['scope']=='source_and_destination' and applied['transforms_destination']
    assert applied['transformed_bodies']==names and applied['fixed_room_free_bodies']==1
    rows={}
    for line in result.stdout.splitlines():
        phase,body,*values=line.split();rows[int(phase),body]=np.array(values,float)
    np.testing.assert_allclose(rows[2,'room_destination'][:3],[-1.8,-.3,.8],atol=1e-12)
    np.testing.assert_allclose(rows[2,'room_extra'][:3],[-2.,-1.,.3],atol=1e-12)
    np.testing.assert_allclose(rows[3,'room_destination'][:3],[-1.73,-.3,.8],atol=1e-12)
    np.testing.assert_allclose(rows[4,'room_destination'][:3],[.4,2.5,.8],atol=1e-12)


@pytest.mark.parametrize('scope,selection',[('unknown','\n'.join(SOURCE)),('source_and_destination',None)])
def test_native_rejects_unknown_or_unselected_explicit_scope(native,tmp_path,scope,selection):
    result=run_native(native,tmp_path,selection,short=True,scope=scope)
    assert result.returncode!=0 and 'room staging selector' in result.stderr
