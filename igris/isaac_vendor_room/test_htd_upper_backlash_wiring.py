"""Explicit upper14 startup selection with unchanged controller/guard contracts."""
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pytest
import transport_trial as trial
import htd_loaded_setup as setup
import htd_loaded as loaded
import htd_loaded_runtime as worker
import htd_backlash_limits as backlash
import run as native
from test_htd_loaded_runtime import Node,Service

OPTIONS=dict(enabled=True,current_grasp_reference=True,command_observer=True,
    wrist_mode_entry=True,goal_hold=True,contact_grace=True,transport=True)
TRIAL_SCOPE=dict(htd_loaded_zero=True,htd_current_grasp_reference=True,htd_command_observer=True,
    htd_wrist_mode_entry=True,htd_loaded_goal_hold=True,htd_contact_grace=True,
    htd_loaded_transport=True,relaxed_grasp=True)


def selection_manifest():
    descriptor=backlash.upper_descriptor(.005)
    return dict(hardware_executed=False,physics_backend='vendor MuJoCo 3.2.7',
        evaluation_profile='relaxed_transport',htd_upper_backlash_limit=descriptor,
        htd_loaded_zero=dict(enabled=True,scope='loaded_transport',current_grasp_reference=True,
            goal_hold={},command_observer={},wrist_mode_entry={},grasp_retention={},upper_backlash_limit=copy.deepcopy(descriptor)))


def test_default_noop_and_no_implicit_selection(tmp_path):
    assert inspect.signature(trial.run).parameters['htd_upper_backlash_timeconst'].default is None
    assert inspect.signature(setup.prepare).parameters['upper_backlash_timeconst'].default is None
    assert setup.prepare(tmp_path/'absent') is None
    assert loaded.upper_backlash_selection({}) is None
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('bad',[True,False,0,1,.02,.01,.0051,float('nan'),float('inf'),'0.005',10**400])
def test_host_and_setup_invalid_value_reject_before_inputs(tmp_path,bad):
    out=tmp_path/'absent'
    with pytest.raises(ValueError,match='upper.*backlash|Upper.*backlash'):
        trial.run(out,tmp_path/'missing',htd_upper_backlash_timeconst=bad)
    with pytest.raises(ValueError,match='upper.*backlash|Upper.*backlash'):
        setup.prepare(out,upper_backlash_timeconst=bad)
    assert not out.exists()


@pytest.mark.parametrize('key',list(TRIAL_SCOPE))
def test_host_requires_every_dependency_explicit_true_before_inputs(tmp_path,key):
    for bad in (False,1):
        options={**TRIAL_SCOPE,key:bad};out=tmp_path/'absent'
        with pytest.raises(ValueError,match='upper.*backlash|Upper.*backlash'):
            trial.run(out,tmp_path/'missing',htd_upper_backlash_timeconst=.005,**options)
        assert not out.exists()


@pytest.mark.parametrize('key',list(OPTIONS))
def test_setup_requires_every_dependency_before_inputs(tmp_path,key):
    for bad in (False,1):
        options={**OPTIONS,key:bad};out=tmp_path/'absent'
        with pytest.raises(ValueError,match='upper.*backlash|Upper.*backlash'):
            setup.prepare(out,upper_backlash_timeconst=.005,**options)
        assert not out.exists()


def test_exact_selection_agreement():
    assert loaded.upper_backlash_selection(selection_manifest())==.005


@pytest.mark.parametrize('bad',['only_top','only_scope','null_top','false_scope','bool_descriptor','float_indices','other_value','wrong_names','upper_ownership','hardware','backend','profile','disabled','not_transport','not_current','observer','wrist','retention','goal'])
def test_manifest_selection_rejects_incomplete_or_changed_descriptor(bad):
    m=selection_manifest();scope=m['htd_loaded_zero']
    if bad=='only_top':scope.pop('upper_backlash_limit')
    if bad=='only_scope':m.pop('htd_upper_backlash_limit')
    if bad=='null_top':m['htd_upper_backlash_limit']=None
    if bad=='false_scope':scope['upper_backlash_limit']=False
    if bad=='bool_descriptor':
        m['htd_upper_backlash_limit']['damping_ratio']=True;scope['upper_backlash_limit']['damping_ratio']=True
    if bad=='float_indices':
        m['htd_upper_backlash_limit']['sdk_indices']=[float(i)for i in range(15,29)];scope['upper_backlash_limit']['sdk_indices']=[float(i)for i in range(15,29)]
    if bad in ('other_value','wrong_names','upper_ownership'):
        key,value={'other_value':('timeconst_s',.02),'wrong_names':('joint_names',['waist_yaw_backlash']),'upper_ownership':('policy_owned',True)}[bad]
        m['htd_upper_backlash_limit'][key]=value;scope['upper_backlash_limit'][key]=value
    if bad=='hardware':m['hardware_executed']=True
    if bad=='backend':m['physics_backend']='other'
    if bad=='profile':m['evaluation_profile']='strict'
    if bad=='disabled':scope['enabled']=False
    if bad=='not_transport':scope['scope']='loaded_zero_only'
    if bad=='not_current':scope['current_grasp_reference']=1
    if bad in ('observer','wrist','retention','goal'):scope.pop({'observer':'command_observer','wrist':'wrist_mode_entry','retention':'grasp_retention','goal':'goal_hold'}[bad])
    with pytest.raises(ValueError,match='upper.*backlash|Upper.*backlash'):loaded.upper_backlash_selection(m)


@pytest.mark.parametrize('cli,expected',[([],None),(['--htd-upper-backlash-timeconst','.005'],.005)])
def test_cli_passes_exact_optional_rate(monkeypatch,tmp_path,cli,expected):
    entry=ast.parse(Path(trial.__file__).read_text()).body[-1];assert isinstance(entry,ast.If)
    calls=[]
    def capture(*args,**kwargs):calls.append(inspect.signature(trial.run).bind(*args,**kwargs).arguments)
    monkeypatch.setattr(sys,'argv',[trial.__file__,'--out',str(tmp_path/'unused'),'--motion',str(tmp_path/'missing'),*cli])
    exec(compile(ast.Module(body=entry.body,type_ignores=[]),trial.__file__,'exec'),dict(vars(trial),run=capture))
    assert len(calls)==1 and calls[0]['htd_upper_backlash_timeconst']==expected


def test_cli_rejects_unscoped_candidate_before_output(tmp_path):
    out=tmp_path/'absent';r=subprocess.run([sys.executable,trial.__file__,'--out',str(out),'--motion',str(tmp_path/'missing'),'--htd-upper-backlash-timeconst','.005'],capture_output=True,text=True)
    assert r.returncode!=0 and 'unrecognized arguments'not in r.stderr and 'backlash'in r.stderr
    assert not out.exists()


@pytest.fixture(scope='module',params=[None,.005],ids=['owned15','owned15_upper14'])
def prepared_trial(request):
    with tempfile.TemporaryDirectory(prefix='htd_upper_wiring_',dir=setup.ROOT/'outputs')as directory:
        out=Path(directory)
        for name in ('manifest.json','robot.yaml','scene.xml','vendor.xml'):
            shutil.copyfile(setup.P/'transfer_trial76/run'/name,out/name)
        path=out/'manifest.json';manifest=json.loads(path.read_text())
        manifest['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
            world_position_lowering=True,torso_relative_lowering=True)
        manifest['evaluation_profile']='relaxed_transport';path.write_text(json.dumps(manifest))
        setup.prepare(out,upper_backlash_timeconst=request.param,**OPTIONS)
        (out/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',hardware_executed=False,simulation_rate_limit=.5)))
        yield out,request.param


def check_start_and_worker(out,monkeypatch):
    manifest=json.loads((out/'manifest.json').read_text());scope=manifest['htd_loaded_zero']
    monkeypatch.setenv('IGRIS_ROOM_PROFILE_LIBRARY',scope['observer_library'])
    env=native.loaded_start_environment(out,manifest)
    node=Node();obj=worker.LoadedRuntime(out,node,services={'mode':Service,'sim':Service},spin=lambda:None)
    assert node.requests==[] and obj.coordinator.policy._step_index==0
    return env,obj


def test_actual_setup_start_worker_match_selection_and_pin_chain(prepared_trial,monkeypatch):
    out,selected=prepared_trial;manifest=json.loads((out/'manifest.json').read_text());scope=manifest['htd_loaded_zero']
    assert loaded.upper_backlash_selection(manifest)==selected
    report=backlash.validate(out,expected_upper_timeconst=selected)
    assert report['owned_sdk_indices']==list(range(15))
    assert report['native_audit']['changed_joints']==(29 if selected is not None else 15)
    for name in ('scene.xml','htd_backlash_original_scene.xml','htd_backlash_limit_audit.json','htd_backlash_audit'):
        assert scope['setup_sha256'][str(out/name)]==setup.sha(out/name)
    for name in ('htd_backlash_limits.py','htd_backlash_limits.cpp'):
        assert scope['setup_sha256'][str(setup.HERE/name)]==setup.sha(setup.HERE/name)
    env,obj=check_start_and_worker(out,monkeypatch)
    assert 'IGRIS_ROOM_SLOW_REALTIME_FACTOR=.1' in env
    assert obj.coordinator.policy.dt==.02 and obj.coordinator.serial_pd.dt==.005
    if selected is None:
        assert 'htd_upper_backlash_limit' not in manifest and 'upper_backlash_limit' not in scope
        assert 'upper_backlash_limit'not in obj.preflight
    else:assert obj.preflight['upper_backlash_limit']==backlash.upper_descriptor(.005)


@pytest.mark.parametrize('tamper',['declare_inverse','changed_count','changed_names','original_hash','final_hash','timeconst','native_version','equality_claim'])
def test_independent_prestart_worker_reject_re_pinned_selection_or_report(prepared_trial,monkeypatch,tamper):
    out,selected=prepared_trial;mp=out/'manifest.json';ap=out/'htd_backlash_limit_audit.json'
    mb,ab=mp.read_bytes(),ap.read_bytes();manifest=json.loads(mb);scope=manifest['htd_loaded_zero'];audit=json.loads(ab)
    if tamper=='declare_inverse':
        if selected is None:
            manifest['htd_upper_backlash_limit']=backlash.upper_descriptor(.005);scope['upper_backlash_limit']=backlash.upper_descriptor(.005)
        else:manifest.pop('htd_upper_backlash_limit');scope.pop('upper_backlash_limit')
    elif tamper=='changed_count':audit['native_audit']['changed_joints']=28
    elif tamper=='changed_names':audit['native_audit']['changed_joint_names']=['waist_yaw_backlash']
    elif tamper=='original_hash':audit['original_scene_sha256']='0'*64
    elif tamper=='final_hash':audit['variant_scene_sha256']='0'*64
    elif tamper=='timeconst':audit['timeconst_s']=.02
    elif tamper=='native_version':audit['native_audit']['mujoco_version']='3.4.0'
    elif tamper=='equality_claim':audit['native_audit']['all_other_serialized_model_fields_equal']=False
    try:
        ap.write_text(json.dumps(audit));scope['setup_sha256'][str(ap)]=setup.sha(ap);mp.write_text(json.dumps(manifest))
        monkeypatch.setenv('IGRIS_ROOM_PROFILE_LIBRARY',scope['observer_library'])
        with pytest.raises((ValueError,RuntimeError)):native.loaded_start_environment(out,manifest)
        class NoServices:
            def get_service_names_and_types(self):raise AssertionError('No service discovery for rejected model')
        with pytest.raises((ValueError,RuntimeError)):
            worker.LoadedRuntime(out,NoServices(),services={'mode':Service,'sim':Service})
    finally:mp.write_bytes(mb);ap.write_bytes(ab)


def test_missing_audit_rejected_even_default_startup(prepared_trial,monkeypatch):
    out,_=prepared_trial;path=out/'htd_backlash_limit_audit.json';original=path.read_bytes()
    manifest=json.loads((out/'manifest.json').read_text());scope=manifest['htd_loaded_zero']
    monkeypatch.setenv('IGRIS_ROOM_PROFILE_LIBRARY',scope['observer_library'])
    try:
        path.unlink()
        with pytest.raises((FileNotFoundError,ValueError)):native.loaded_start_environment(out,manifest)
        with pytest.raises((FileNotFoundError,ValueError)):worker.LoadedRuntime(out,Node(),services={'mode':Service,'sim':Service})
    finally:path.write_bytes(original)
