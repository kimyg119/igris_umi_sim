"""Trial-local passive DIP damping, native compilation only; no physics."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/hlik_transport_20260918_2005/transfer_trial76/run'


@pytest.fixture
def trial():
    with tempfile.TemporaryDirectory(prefix='dip_damping_test_',dir=ROOT/'outputs') as folder:
        path=Path(folder)
        for name in ('scene.xml','vendor.xml','manufacturer.xml','compliance.json','manifest.json','robot.yaml'):
            shutil.copyfile(SOURCE/name,path/name)
        manifest=json.loads((path/'manifest.json').read_text())
        manifest.update(dip_damping_requested=.002,dip_damping_scope='loaded_transport')
        manifest['walking_transfer'].update(xy_first_lowering=True,clearance_minimum_rise_m=.02)
        (path/'manifest.json').write_text(json.dumps(manifest))
        yield path


def test_none_does_not_read_or_mutate_anything(tmp_path):
    from dip_damping import configure
    assert configure(tmp_path/'absent',None) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('value',[.0001,.001,0.,-.002,float('nan'),float('inf'),True])
def test_only_reviewed_coefficient_is_accepted_before_io(tmp_path,value):
    from dip_damping import configure
    with pytest.raises(ValueError):configure(tmp_path,value)
    assert not list(tmp_path.iterdir())


def test_baseline_absence_needs_no_new_artifacts(trial):
    from dip_damping import validate
    path=trial/'manifest.json';m=json.loads(path.read_text())
    del m['dip_damping_requested'];del m['dip_damping_scope'];path.write_text(json.dumps(m))
    before={p.name:p.read_bytes() for p in trial.iterdir()}
    assert validate(trial) is None
    assert before=={p.name:p.read_bytes() for p in trial.iterdir()}


def test_unselected_baseline_outside_project_remains_inert(tmp_path):
    from dip_damping import validate
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    assert validate(tmp_path) is None
    assert manifest.read_text()=='{}'
    assert list(tmp_path.iterdir())==[manifest]


@pytest.mark.parametrize('name',['dip_damping_audit','dip_damping_candidate_scene.xml'])
def test_orphan_audit_or_candidate_rejected_without_descriptor(trial,name):
    from dip_damping import validate
    manifest=trial/'manifest.json';data=json.loads(manifest.read_text())
    del data['dip_damping_requested'];del data['dip_damping_scope']
    manifest.write_text(json.dumps(data));(trial/name).write_text('orphan')
    with pytest.raises(ValueError,match='descriptor'):validate(trial)


@pytest.mark.parametrize('change',[{'hardware_executed':True},{'dip_damping_requested':.001},
                                 {'dip_damping_scope':'loaded_goal_hold_only'},
                                 {'evaluation_profile':'original_grasp_requirements'}])
def test_bad_explicit_scope_rejected_without_model_changes(trial,change):
    from dip_damping import configure
    path=trial/'manifest.json';m=json.loads(path.read_text());m.update(change);path.write_text(json.dumps(m))
    before={name:(trial/name).read_bytes() for name in ('scene.xml','vendor.xml','compliance.json')}
    with pytest.raises(ValueError):configure(trial,.002)
    assert all((trial/name).read_bytes()==data for name,data in before.items())
    assert not (trial/'dip_damping_original_scene.xml').exists()


@pytest.fixture(scope='module')
def configured():
    from dip_damping import configure
    with tempfile.TemporaryDirectory(prefix='dip_damping_configured_',dir=ROOT/'outputs') as folder:
        path=Path(folder)
        for name in ('scene.xml','vendor.xml','manufacturer.xml','compliance.json','manifest.json','robot.yaml'):
            shutil.copyfile(SOURCE/name,path/name)
        m=json.loads((path/'manifest.json').read_text())
        m.update(dip_damping_requested=.002,dip_damping_scope='loaded_transport')
        m['walking_transfer'].update(xy_first_lowering=True,clearance_minimum_rise_m=.02)
        (path/'manifest.json').write_text(json.dumps(m))
        before={p.name:p.read_bytes() for p in path.iterdir()}
        result=configure(path,.002)
        yield path,before,result


def test_native_compilation_changes_only_two_dip_damping_fields_in_both_models(configured):
    from dip_damping import validate,configure
    path,before,result=configured
    assert validate(path)==result
    assert (path/'manufacturer.xml').read_bytes()==before['manufacturer.xml']
    for name in ('vendor.xml','scene.xml'):
        assert (path/('dip_damping_original_'+name)).read_bytes()==before[name]
        audit=result['native_audits'][name]
        assert audit['physics_steps']==0 and audit['mujoco_version']=='3.2.7'
        assert audit['changed_dofs']==2 and audit['all_other_serialized_model_fields_equal']
        assert set(audit['changed_joint_names'])=={'l_hand_dip_spring','r_hand_dip_spring'}
        original=ET.fromstring(before[name]);current=ET.parse(path/name).getroot()
        for side in ('l','r'):
            j=current.find(f".//joint[@name='{side}_hand_dip_spring']")
            assert float(j.get('damping'))==.002
            j.set('damping',original.find(f".//joint[@name='{side}_hand_dip_spring']").get('damping'))
        assert ET.canonicalize(ET.tostring(original,encoding='unicode'),strip_text=True)==ET.canonicalize(ET.tostring(current,encoding='unicode'),strip_text=True)
    compliance=json.loads((path/'compliance.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    assert compliance['dip_spring']['damping_Nms_rad']==.002
    assert manifest['compliant_hands']==compliance
    with pytest.raises(FileExistsError):configure(path,.002)


@pytest.mark.parametrize('field',['damping','stiffness','springref','range','mcp','mass','equality'])
def test_native_auditor_rejects_any_additional_physical_change(configured,tmp_path,field):
    path,_,_=configured;tree=ET.parse(path/'scene.xml')
    dip=tree.find(".//joint[@name='l_hand_dip_spring']")
    if field=='mcp':tree.find(".//joint[@name='l_hand_mcp_compliant']").set('damping','.003')
    elif field=='mass':tree.find(".//body[@name='l_hand_dip']/inertial").set('mass','.05')
    elif field=='equality':tree.find('./equality/connect').set('solref','.003 1')
    else:dip.set(field,{'damping':'.003','stiffness':'.05','springref':'.01','range':'-.5 .5'}[field])
    candidate=tmp_path/'tamper.xml';tree.write(candidate)
    run=subprocess.run([str(path/'dip_damping_audit'),str(path/'dip_damping_original_scene.xml'),str(candidate)],capture_output=True,text=True)
    assert run.returncode!=0


@pytest.mark.parametrize('name',['scene.xml','vendor.xml','compliance.json','dip_damping.json',
                                'dip_damping_original_scene.xml'])
def test_validation_rejects_changed_artifact_or_metadata(configured,name):
    from dip_damping import validate
    path,_,_=configured;file=path/name;before=file.read_bytes()
    file.write_bytes(before+b' ')
    try:
        with pytest.raises(ValueError):validate(path)
    finally:file.write_bytes(before)


def test_validation_rejects_dropped_optin_or_wrong_actual_htd_scope(configured):
    from dip_damping import validate
    path,_,_=configured;file=path/'manifest.json';before=file.read_bytes()
    try:
        m=json.loads(before);m.pop('dip_damping_requested');file.write_text(json.dumps(m))
        with pytest.raises(ValueError):validate(path)
        m=json.loads(before);m['htd_loaded_zero']={'enabled':True,'scope':'loaded_goal_hold_only'}
        file.write_text(json.dumps(m))
        with pytest.raises(ValueError):validate(path)
    finally:file.write_bytes(before)


def test_loaded_setup_pins_variant_and_worker_independently_validates(trial):
    import htd_loaded_setup as setup
    import htd_loaded_runtime as runtime
    from dip_damping import configure,validate
    from test_htd_loaded_runtime import Node,Service
    descriptor=configure(trial,.002)
    evidence=setup.prepare(trial,enabled=True,current_grasp_reference=True,command_observer=True,
        wrist_mode_entry=True,goal_hold=True,contact_grace=True,transport=True,environment={})
    assert validate(trial)==descriptor
    scope=evidence['manifest_scope'];assert scope['dip_damping']==descriptor
    for name in ('compliance.json','dip_damping.json','dip_damping_audit',
                 'dip_damping_original_vendor.xml','dip_damping_original_scene.xml',
                 'htd_backlash_original_scene.xml','htd_backlash_limit_audit.json'):
        assert str(trial/name) in scope['setup_sha256']
    (trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',hardware_executed=False,simulation_rate_limit=.5)))
    obj=runtime.LoadedRuntime(trial,Node(),services={'mode':Service,'sim':Service},spin=lambda:None)
    assert obj.coordinator.transport and obj.coordinator.policy._step_index==0
    m=json.loads((trial/'manifest.json').read_text());m['dip_damping_requested']=.001
    (trial/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):runtime.LoadedRuntime(trial,Node(),services={'mode':Service,'sim':Service},spin=lambda:None)


def test_loaded_setup_refuses_variant_outside_transport_before_mutation(trial):
    from dip_damping import configure
    import htd_loaded_setup as setup
    configure(trial,.002)
    with pytest.raises(ValueError,match='DIP|dip'):
        setup.prepare(trial,enabled=True,environment={})
    assert not (trial/'htd_gc_params.yaml').exists()
    assert not (trial/'htd_backlash_original_scene.xml').exists()
