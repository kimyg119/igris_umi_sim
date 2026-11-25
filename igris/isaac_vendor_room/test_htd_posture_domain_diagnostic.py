"""Explicit out-of-profile SIM diagnostic retains measured handoff guards."""
from dataclasses import replace
from types import SimpleNamespace as NS
import copy,inspect,json,subprocess,sys
import numpy as np
import pytest
import htd_candidate_package as candidate
import htd_loaded_setup as host
import htd_loaded_runtime as runtime
import transport_trial
from test_htd_candidate_package import package
from test_htd_posture_candidate import posture_package
from test_htd_loaded_setup import fresh_trial
from test_htd_loaded_runtime import Node,Service
from test_htd_loaded_transport import coordinator,sample

SELECTION={'selection':'evaluated_posture_continuation','profile_handoff_max_error_rad':.1}


def test_default_strict_and_diagnostic_preserves_original_criterion_and_full_delta():
    hold=np.zeros(31);actual=hold.copy();actual[[22,23,24,25]]=[-.637,-.192,.7505613188861642,.116]
    with pytest.raises(ValueError,match='0.1rad'):candidate.check_posture_handoff(SELECTION,hold,actual)
    before=(hold.copy(),actual.copy())
    result=candidate.check_posture_handoff(SELECTION,hold,actual,posture_domain_diagnostic=True)
    assert result['profile_match'] is False and result['limit_rad']==.1
    assert result['posture_domain_diagnostic'] is True and result['runtime_domain_validated'] is False
    assert result['delta_rad']==actual[[15,16,17,18,22,23,24,25]].tolist()
    assert result['max_abs_error_rad']==.7505613188861642
    assert np.array_equal(hold,before[0])and np.array_equal(actual,before[1])
    matched=candidate.check_posture_handoff(SELECTION,hold,hold,posture_domain_diagnostic=True)
    assert matched['profile_match'] is True


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_malformed_boolean_rejects_even_before_noop_or_legacy_branch(value):
    with pytest.raises(ValueError,match='bool'):
        candidate.check_posture_handoff({'selection':'evaluated_continuation'},np.zeros(31),np.zeros(31),posture_domain_diagnostic=value)


@pytest.mark.parametrize('bad',[np.zeros(30),np.full(31,np.nan),np.full(31,np.inf)])
def test_diagnostic_does_not_bypass_finite31_contract(bad):
    with pytest.raises(ValueError,match='Finite31'):
        candidate.check_posture_handoff(SELECTION,np.zeros(31),bad,posture_domain_diagnostic=True)


def test_diagnostic_rejects_overflowing_finite_reference_delta():
    hold=np.zeros(31);actual=hold.copy();hold[24]=-1e308;actual[24]=1e308
    with pytest.raises(ValueError,match='Finite|finite'):
        candidate.check_posture_handoff(SELECTION,hold,actual,posture_domain_diagnostic=True)


@pytest.mark.parametrize('kind,transport,current',[('frozen1750_profile76',True,True),('evaluated_continuation',True,True),
    ('evaluated_posture_continuation',False,True),('evaluated_posture_continuation',True,False)])
def test_scope_rejects_legacy_or_nontransport_or_uncaptured_grasp(kind,transport,current):
    assert hasattr(candidate,'require_posture_domain_diagnostic')
    with pytest.raises(ValueError,match='diagnostic'):
        candidate.require_posture_domain_diagnostic(True,{**SELECTION,'selection':kind},transport=transport,current_grasp_reference=current)


def test_launcher_option_defaults_false_and_rejects_scope_before_output_creation(tmp_path):
    assert inspect.signature(transport_trial.run).parameters['htd_posture_domain_diagnostic'].default is False
    out=tmp_path/'uncreated'
    process=subprocess.run([sys.executable,str(host.HERE/'transport_trial.py'),'--out',str(out),
        '--motion',str(tmp_path/'absent_motion'),'--htd-posture-domain-diagnostic'],capture_output=True,text=True)
    assert process.returncode!=0 and 'diagnostic' in process.stderr and 'requires' in process.stderr
    assert 'unrecognized arguments' not in process.stderr and not out.exists()


def runtime_stub(diagnostic=True):
    c,p,_=coordinator();obj=runtime.LoadedRuntime.__new__(runtime.LoadedRuntime)
    obj.policy_selection=SELECTION.copy();obj.posture_domain_diagnostic=diagnostic;obj.events=[]
    obj.coordinator=c;obj.entry_ready=lambda s:True;obj._services={'mode':Service}
    node=Node();obj._clients={'mode':node.create_client(Service,'mode')}
    s=replace(sample(p,mode=6),phase_target_valid=True)
    upper=s.upper_output_target.copy();upper[24]+=.75
    output=copy.deepcopy(s.output);output['q_output'][15:29]=upper[15:29]
    output['q_encoder'][15:29]=upper[15:29]
    s=replace(s,upper_output_target=upper,output=output)
    return obj,s,node


def test_runtime_out_of_domain_event_is_not_qualified_and_captures_actual_reference_once():
    obj,s,node=runtime_stub();obj.request_low(s)
    assert len(node.requests)==1
    assert obj.events[0]['event']=='posture_handoff_out_of_domain_diagnostic'
    assert obj.events[0]['profile_match'] is False
    assert not any(x['event']=='posture_handoff_qualified'for x in obj.events)
    evidence=obj.coordinator.current_grasp_evidence
    assert evidence['max_abs_delta_from_frozen_profile_rad']==pytest.approx(.75)
    assert evidence['accepted_target_rad']==s.upper_output_target[[15,16,17,18,22,23,24,25]].tolist()
    later=s.upper_output_target.copy();later[24]+=.1001
    with pytest.raises(RuntimeError,match='captured current grasp'):
        obj.coordinator._check(replace(s,upper_output_target=later))


def test_default_runtime_still_rejects_before_low_or_reference_capture():
    obj,s,node=runtime_stub(False)
    with pytest.raises(ValueError,match='0.1rad'):obj.request_low(s)
    assert node.requests==[] and obj.coordinator.current_grasp_evidence is None
    assert obj.events[-1]['event']=='posture_handoff_rejected'


@pytest.mark.parametrize('change',['physical_upper','hardware','weak_hand','stale','fallen'])
def test_diagnostic_keeps_all_actual_handoff_checks_before_low(change):
    obj,s,node=runtime_stub()
    if change=='physical_upper':s.output['q_output'][24]+=.201
    elif change=='hardware':s=replace(s,environment=0)
    elif change=='weak_hand':s=replace(s,palm_tip_force_N=np.zeros(4))
    elif change=='stale':s=replace(s,hl_age_s=.25)
    else:s=replace(s,pelvis_pose=np.array([0,0,.69,1,0,0,0]))
    reason={'physical_upper':'Physical upper14','hardware':'SIM RobotState','weak_hand':'grasp',
            'stale':'hl_age','fallen':'fall guard'}[change]
    with pytest.raises((ValueError,RuntimeError),match=reason):obj.request_low(s)
    assert node.requests==[] and obj.coordinator.state=='prepared'


def prepare(trial,directory,diagnostic=True,transport=True):
    path=trial/'manifest.json';manifest=json.loads(path.read_text())
    manifest['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
        world_position_lowering=True,torso_relative_lowering=True)
    path.write_text(json.dumps(manifest))
    return host.prepare(trial,enabled=True,current_grasp_reference=True,command_observer=True,
        wrist_mode_entry=True,goal_hold=True,contact_grace=True,transport=transport,policy_package=directory,
        posture_domain_diagnostic=diagnostic)


def test_host_and_worker_independently_record_valid_optin_and_reject_malformed(fresh_trial,posture_package):
    directory,_=posture_package;prepared=prepare(fresh_trial,directory)
    scope=prepared['manifest_scope'];assert scope['posture_domain_diagnostic'] is True
    assert scope['profile_handoff_max_error_rad']==.1
    assert scope['runtime_domain_validated'] is False
    (fresh_trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',hardware_executed=False,simulation_rate_limit=.5)))
    node=Node();obj=runtime.LoadedRuntime(fresh_trial,node,services={'mode':Service,'sim':Service},spin=lambda:None)
    assert obj.posture_domain_diagnostic is True and obj.preflight['posture_domain_diagnostic'] is True
    assert obj.preflight['runtime_domain_validated'] is False and node.requests==[]
    path=fresh_trial/'manifest.json';original=path.read_bytes();manifest=json.loads(original)
    manifest['htd_loaded_zero']['posture_domain_diagnostic']='true';path.write_text(json.dumps(manifest))
    class NoServices:
        def get_service_names_and_types(self):raise AssertionError('No discovery on invalid diagnostic metadata')
    try:
        with pytest.raises(ValueError,match='bool'):runtime.LoadedRuntime(fresh_trial,NoServices(),services={'mode':Service,'sim':Service})
    finally:path.write_bytes(original)


@pytest.mark.parametrize('bad',['legacy','nontransport','nonbool'])
def test_host_invalid_optin_never_mutates_generated_scene(fresh_trial,request,bad):
    names=['scene.xml','vendor.xml','robot.yaml'];before={n:(fresh_trial/n).read_bytes()for n in names}
    directory=request.getfixturevalue('package'if bad=='legacy'else'posture_package')[0]
    with pytest.raises(ValueError,match='diagnostic|bool'):
        prepare(fresh_trial,directory,diagnostic='true'if bad=='nonbool'else True,transport=bad!='nontransport')
    assert all((fresh_trial/n).read_bytes()==v for n,v in before.items())
    assert not (fresh_trial/'htd_backlash_original_scene.xml').exists()
