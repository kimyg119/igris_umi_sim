"""Actual host/worker package wiring; no physics, mode requests or training."""
import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest

from htd_loaded import LoadedZero
import htd_loaded_setup as host
from htd_loaded_runtime import LoadedRuntime
from test_htd_loaded_setup import fresh_trial
from test_htd_candidate_package import package
from test_htd_posture_candidate import posture_package
from test_htd_loaded_runtime import Node, Service
from test_htd_loaded_transport import coordinator


class NoServices:
    def get_service_names_and_types(self):
        raise AssertionError('No discovery permitted after invalid reviewed approach selection')


def prepare(trial,directory,transport):
    path=trial/'manifest.json';manifest=json.loads(path.read_text())
    manifest['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
        world_position_lowering=True,torso_relative_lowering=True)
    path.write_text(json.dumps(manifest))
    evidence=host.prepare(trial,enabled=True,current_grasp_reference=True,command_observer=True,
        wrist_mode_entry=True,goal_hold=True,contact_grace=True,transport=transport,policy_package=directory)
    (trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',
        hardware_executed=False,simulation_rate_limit=.5)))
    return evidence


def worker(trial,node=None):
    return LoadedRuntime(trial,node or Node(),services={'mode':Service,'sim':Service},spin=lambda:None)


def test_coordinator_option_defaults_false_and_true_requires_transport():
    assert inspect.signature(LoadedZero).parameters['reviewed_approach'].default is False
    with pytest.raises(ValueError,match='reviewed.*transport|Reviewed.*transport'):
        LoadedZero(None,None,np.zeros(31),mass_kg=.1,friction=4.,clearance_lift_m=.04,
                   reviewed_approach=True)
    obj,_,_=coordinator(reviewed_approach=True)
    assert obj.reviewed_approach and obj.goal_hold.reviewed_approach
    assert obj.contract['reviewed_approach'] is True


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_coordinator_rejects_non_bool_reviewed_option_before_policy_access(value):
    with pytest.raises(ValueError,match='boolean|bool'):
        LoadedZero(None,None,np.zeros(31),mass_kg=.1,friction=4.,clearance_lift_m=.04,
                   reviewed_approach=value)


@pytest.mark.parametrize('kind,transport,expected',[
    ('frozen',True,False),('legacy',True,False),('posture',False,False),('posture',True,True)])
def test_host_worker_only_enable_reviewed_route_for_explicit_posture_transport(
        request,fresh_trial,kind,transport,expected):
    directory=None
    if kind=='legacy':directory=request.getfixturevalue('package')[0]
    elif kind=='posture':directory=request.getfixturevalue('posture_package')[0]
    evidence=prepare(fresh_trial,directory,transport);scope=evidence['manifest_scope']
    assert scope.get('transport',{}).get('reviewed_approach',False)is expected
    assert ('reviewed_approach'in scope['goal_hold'])is expected
    node=Node();obj=worker(fresh_trial,node)
    assert obj.coordinator.reviewed_approach is expected
    assert obj.coordinator.goal_hold.contract==scope['goal_hold']
    assert obj.preflight.get('reviewed_approach',False)is expected
    assert obj.coordinator.policy._step_index==0 and node.requests==[]
    if expected:
        source=host.HERE/'htd_reviewed_approach.py'
        assert scope['setup_sha256'][str(source)]==host.sha(source)
        assert scope['goal_hold']['reviewed_approach']['geometry_review']['dynamic_validation']=='pending'
    if transport:
        # Reinterpreting a validated package's route selection fails before services.
        path=fresh_trial/'manifest.json';saved=path.read_bytes();manifest=json.loads(saved)
        manifest['htd_loaded_zero']['transport']['reviewed_approach']=not expected
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='reviewed.approach|Reviewed.approach'):
            worker(fresh_trial,NoServices())
        path.write_bytes(saved)


def test_worker_rejects_removed_or_nonbool_flag_and_missing_source_or_geometry_pin(fresh_trial,posture_package):
    prepare(fresh_trial,posture_package[0],True)
    path=fresh_trial/'manifest.json';saved=path.read_bytes();original=json.loads(saved)
    for change in ('flag_missing','flag_nonbool','source_missing','source_hash','geometry_pin'):
        manifest=copy.deepcopy(original);scope=manifest['htd_loaded_zero']
        if change=='flag_missing':scope['transport'].pop('reviewed_approach',None)
        elif change=='flag_nonbool':scope['transport']['reviewed_approach']=1
        elif change=='source_missing':scope['setup_sha256'].pop(str(host.HERE/'htd_reviewed_approach.py'),None)
        elif change=='source_hash':scope['setup_sha256'][str(host.HERE/'htd_reviewed_approach.py')]='0'*64
        else:scope['goal_hold']['reviewed_approach']['geometry_review']['report_sha256']='0'*64
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='reviewed.approach|Reviewed.approach|hash|goal hold'):
            worker(fresh_trial,NoServices())
    path.write_bytes(saved)
