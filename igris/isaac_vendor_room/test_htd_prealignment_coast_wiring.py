"""Coast admission/configuration before IO; no actor inference or native physics."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import htd_loaded as loaded
import htd_loaded_setup as setup
import htd_loaded_runtime as runtime
import transport_trial
import run
from htd_reviewed_approach import coast_descriptor,COAST_ARTIFACTS,COAST_REPORT,COAST_REPORT_SHA256

ROOT=Path(__file__).resolve().parents[2]
FLAGS=dict(enabled=True,transport=True,current_grasp_reference=True,command_observer=True,
           wrist_mode_entry=True,goal_hold=True,contact_grace=True,relaxed_grasp=True)


def selection():
    return dict(selection='evaluated_posture_continuation',artifact_sha256=dict(COAST_ARTIFACTS))


def manifest():
    descriptor=coast_descriptor(3.)
    return dict(hardware_executed=False,physics_backend='vendor MuJoCo 3.2.7',evaluation_profile='relaxed_transport',
        htd_prealignment_coast=descriptor,
        htd_loaded_zero=dict(enabled=True,scope='loaded_transport',current_grasp_reference=True,
            prealignment_coast=copy.deepcopy(descriptor),goal_hold={'prealignment_coast':copy.deepcopy(descriptor)},
            transport={'reviewed_approach':True},command_observer={},wrist_mode_entry={},grasp_retention={},
            policy_selection=selection(),setup_sha256={str(ROOT/COAST_REPORT):COAST_REPORT_SHA256}))


def test_absence_is_inert_and_selected_descriptor_is_valid():
    assert loaded.prealignment_coast_selection({}) is None
    assert loaded.prealignment_coast_selection(manifest())==3.
    assert loaded.validate_prealignment_coast_option(None,**{k:False for k in FLAGS}) is None
    assert loaded.validate_prealignment_coast_option(3,**FLAGS)==3.


@pytest.mark.parametrize('key',list(FLAGS))
def test_every_explicit_scope_flag_required(key):
    flags=dict(FLAGS);flags[key]=False
    with pytest.raises(ValueError,match='coast'):
        loaded.validate_prealignment_coast_option(3.,**flags)


@pytest.mark.parametrize('bad',[True,'3',0,2,4,float('nan'),float('inf'),10**400])
def test_bad_options_rejected_before_setup_reads_or_writes(tmp_path,bad):
    target=tmp_path/'absent'
    with pytest.raises(ValueError,match='coast'):
        setup.prepare(target,prealignment_coast_seconds=bad)
    assert not target.exists()


@pytest.mark.parametrize('where', ['top','scope','goal','report','policy','transport','simulation','duration_type','bool_type','extra'])
def test_corruption_rejected_independently_by_worker_and_prestart(tmp_path,where):
    m=manifest();s=m['htd_loaded_zero']
    if where=='top': del m['htd_prealignment_coast']
    if where=='scope': del s['prealignment_coast']
    if where=='goal': del s['goal_hold']['prealignment_coast']
    if where=='report': s['setup_sha256'][str(ROOT/COAST_REPORT)]='0'*64
    if where=='policy': s['policy_selection']['artifact_sha256']['policy.ts']='0'*64
    if where=='transport': s['transport']['reviewed_approach']=False
    if where=='simulation': m['hardware_executed']=True
    if where=='duration_type': m['htd_prealignment_coast']['duration_native_s']=3
    if where=='bool_type': m['htd_prealignment_coast']['once']=1
    if where=='extra': s['prealignment_coast']['bypass']=True
    with pytest.raises(ValueError,match='[Cc]oast'): loaded.prealignment_coast_selection(m)
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='[Cc]oast'): runtime.LoadedRuntime(tmp_path,object())
    with pytest.raises(ValueError,match='[Cc]oast'): run.loaded_start_environment(tmp_path,m)


@pytest.mark.parametrize('name',list(COAST_ARTIFACTS))
def test_selected_artifact_mismatch_is_not_new_policy_permission(name):
    s=selection();s['artifact_sha256'][name]='1'*64
    with pytest.raises(ValueError,match='[Cc]oast'):
        loaded.validate_prealignment_coast_policy(s)


def test_coordinator_receives_same_contract_without_invoking_actor():
    policy=SimpleNamespace(dt=.02,policy_sha256=COAST_ARTIFACTS['policy.ts'])
    pd=SimpleNamespace(dt=.005,contract={'configured_delay_steps':[0,0,0,0],'feasible_command_q':np.zeros(31)})
    c=loaded.LoadedZero(policy,pd,np.zeros(31),mass_kg=.01,friction=4.,clearance_lift_m=.05,
        current_grasp_reference=True,goal_hold=True,contact_grace=True,transport_target_xyz=[1,0,.8],
        transport_heading=0.,reviewed_approach=True,prealignment_coast_seconds=3.)
    assert c.goal_hold.contract['prealignment_coast']==coast_descriptor(3.)
    assert c.state=='prepared' and c.actor_count==0 and c.published_count==0
    assert c.contract['actor_dt_s']==.02 and c.contract['pd_dt_s']==.005


def test_cli_exposes_optin_and_default_before_any_launch(monkeypatch):
    tree=ast.parse(Path(transport_trial.__file__).read_text())
    entry=tree.body[-1]
    assert isinstance(entry,ast.If)
    calls=[]
    def capture(*args,**kwargs): calls.append((args,kwargs));return {'offline':True}
    ns=dict(transport_trial.__dict__);ns['run']=capture
    import sys
    for extra,want in [([],None),(['--htd-prealignment-coast-seconds','3'],3.)]:
        monkeypatch.setattr(sys,'argv',['transport_trial','--out','/tmp/unused','--motion','/tmp/unused_motion',*extra])
        exec(compile(ast.Module(body=entry.body,type_ignores=[]),'<CLI test>','exec'),ns)
        assert calls[-1][0][-1]==want


def test_invalid_cli_scope_cannot_create_output(tmp_path):
    out=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='[Cc]oast'):
        transport_trial.run(out,tmp_path/'no_motion',htd_prealignment_coast_seconds=3.)
    assert not out.exists()


def test_real_setup_prestart_worker_forward_same_selection_without_actor_inference(monkeypatch):
    """Use actual artifacts/static model audits, replacing only actor and package boundary.

    Historical package source pins are intentionally not promoted here. Its four
    immutable bytes are independently checked; parent will build the new wrapper.
    """
    import shutil,tempfile
    import htd_policy,htd_serial_pd
    from test_htd_loaded_runtime import Node,Service
    p=setup.P
    selected=json.loads((p/'transfer_trial111/run/manifest.json').read_text())['htd_loaded_zero']['policy_selection']
    assert selected['artifact_sha256']==COAST_ARTIFACTS
    for name,digest in COAST_ARTIFACTS.items():
        assert setup.sha(Path(selected['package_directory'])/name)==digest
    monkeypatch.setattr(setup,'resolve_package',lambda path:copy.deepcopy(selected))
    monkeypatch.setattr(runtime,'validate_runtime_selection',lambda scope:copy.deepcopy(selected))
    class NoInferencePolicy:
        dt=.02
        policy_sha256=COAST_ARTIFACTS['policy.ts']
        _step_index=0
        def __init__(self,*args): pass
        def step(self,*args): raise AssertionError('No policy inference authorized')
    monkeypatch.setattr(htd_policy,'HTDPolicy',NoInferencePolicy)
    monkeypatch.setattr(runtime,'HTDPolicy',NoInferencePolicy)
    monkeypatch.setattr(htd_serial_pd,'warm_policy_actor',lambda policy:None)
    monkeypatch.setattr(runtime,'warm_policy_actor',lambda policy:None)
    with tempfile.TemporaryDirectory(prefix='coast_static_wiring_',dir=ROOT/'outputs') as directory:
        trial=Path(directory)
        for name in ('manifest.json','robot.yaml','scene.xml','vendor.xml'):
            shutil.copyfile(p/'transfer_trial76/run'/name,trial/name)
        m=json.loads((trial/'manifest.json').read_text())
        m.update(hardware_executed=False,physics_backend='vendor MuJoCo 3.2.7',evaluation_profile='relaxed_transport')
        m['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
            world_position_lowering=True,torso_relative_lowering=True)
        (trial/'manifest.json').write_text(json.dumps(m))
        args={k:v for k,v in FLAGS.items() if k!='relaxed_grasp'}
        setup.prepare(trial,policy_package=Path(selected['package_directory']),prealignment_coast_seconds=3.,**args)
        m=json.loads((trial/'manifest.json').read_text());scope=m['htd_loaded_zero']
        assert loaded.prealignment_coast_selection(m)==3.
        for name in ('htd_reviewed_approach.py','htd_navigation_feedback.py','htd_loaded.py','htd_loaded_setup.py',
                     'htd_loaded_runtime.py','transport_trial.py','run.py'):
            path=setup.HERE/name;assert scope['setup_sha256'][str(path)]==setup.sha(path)
        monkeypatch.setenv('IGRIS_ROOM_PROFILE_LIBRARY',scope['observer_library'])
        assert 'IGRIS_ROOM_SLOW_REALTIME_FACTOR=.1' in run.loaded_start_environment(trial,m)
        (trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',hardware_executed=False,simulation_rate_limit=.5)))
        node=Node();worker=runtime.LoadedRuntime(trial,node,services={'mode':Service,'sim':Service},spin=lambda:None)
        assert node.requests==[] and worker.coordinator.actor_count==0
        assert worker.coordinator.goal_hold.contract['prealignment_coast']==coast_descriptor(3.)
        assert worker.preflight['goal_hold']['prealignment_coast']==coast_descriptor(3.)


def test_fresh_setup_rejects_orphan_descriptor_before_any_configuration_write(tmp_path,monkeypatch):
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'htd_prealignment_coast':coast_descriptor(3.)}))
    with pytest.raises(ValueError,match='[Cc]oast'):
        setup.prepare(tmp_path,enabled=True)
    assert sorted(p.name for p in tmp_path.iterdir())==['manifest.json']
