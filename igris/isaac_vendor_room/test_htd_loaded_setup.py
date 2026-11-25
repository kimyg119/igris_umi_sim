"""Host setup only: offline native model audits are allowed, physics is not."""
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest
import yaml

import htd_loaded_setup as setup
import transport_trial
from test_htd_candidate_package import package


def test_opt_in_flag_defaults_false_and_invalid_context_fails_before_preparation(tmp_path):
    assert inspect.signature(transport_trial.run).parameters['htd_loaded_zero'].default is False
    with pytest.raises(ValueError, match='HTD loaded'):
        transport_trial.run(tmp_path/'unused', tmp_path/'absent_motion', htd_loaded_zero=True)
    assert not (tmp_path/'unused').exists()


def test_disabled_setup_does_not_read_inputs_write_files_or_change_environment(tmp_path):
    destination = tmp_path/'absent'
    assert setup.prepare(destination, enabled=False, environment={}) is None
    assert not destination.exists()


def test_runtime_hook_gate_prevents_silent_fallback_to_vendor_walking():
    with pytest.raises(RuntimeError, match='integration'):
        setup.require_runtime_hooks(object())
    class Integrated:
        HTD_LOADED_ZERO_RUNTIME_CONTRACT = 'htd_loaded_zero/v1'
    setup.require_runtime_hooks(Integrated)


@pytest.fixture
def fresh_trial():
    source = setup.ROOT/'outputs/hlik_transport_20260918_2005/transfer_trial76/run'
    if not source.exists() or not setup.PACKAGE.exists():
        pytest.skip('Explicit profile76 and frozen1750 local inputs required for offline integration audit')
    with tempfile.TemporaryDirectory(prefix='htd_loaded_setup_test_', dir=setup.ROOT/'outputs') as temporary:
        trial = Path(temporary)
        for name in ('manifest.json', 'robot.yaml', 'scene.xml', 'vendor.xml'):
            shutil.copyfile(source/name, trial/name)
        yield trial


@pytest.mark.parametrize('name', ['runtime.json', 'physics_stats.json', 'pacing.request',
    'htd_pacing.request', 'htd_pacing.request.ack', 'htd_initial_state.request', 'htd_initial_state_applied.json'])
def test_started_or_activated_directory_is_rejected_before_model_edits(fresh_trial, name):
    scene = (fresh_trial/'scene.xml').read_bytes()
    (fresh_trial/name).write_text('{}')
    with pytest.raises((ValueError, FileExistsError), match='fresh|activation|initializer'):
        setup.prepare(fresh_trial, enabled=True, environment={})
    assert (fresh_trial/'scene.xml').read_bytes() == scene
    assert not (fresh_trial/'htd_gc_params.yaml').exists()


def test_existing_profile_interposer_and_initializer_manifest_are_rejected(fresh_trial):
    with pytest.raises(ValueError, match='interposer'):
        setup.prepare(fresh_trial, enabled=True, environment={'IGRIS_ROOM_PROFILE_LIBRARY': '/do/not/replace.so'})
    manifest = json.loads((fresh_trial/'manifest.json').read_text())
    manifest['htd_initial_state'] = {}
    (fresh_trial/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='initializer'):
        setup.prepare(fresh_trial, enabled=True, environment={})


def test_actual_frozen_package_prepares_scoped_gc_model_observer_and_hashes_without_launch(fresh_trial, monkeypatch):
    original_robot = yaml.safe_load((fresh_trial/'robot.yaml').read_text())
    original_params = yaml.safe_load(setup.PARAMS.read_text())
    original_run = subprocess.run
    executed = []
    def offline_only(command, *args, **kwargs):
        name = Path(command[0]).name
        assert name == 'g++' or name in ('htd_backlash_audit', 'htd_output_audit'), command
        executed.append(name)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', offline_only)
    environment = {'unrelated': 'preserved'}
    before = copy.deepcopy(environment)
    evidence = setup.prepare(fresh_trial, enabled=True, environment=environment)
    assert environment == before
    assert evidence['pass'] and evidence['physics_steps'] == 0 and not evidence['hardware_executed']
    assert evidence['runtime_private_sim_verified'] is False
    assert 'htd_backlash_audit' in executed and 'htd_output_audit' in executed
    manifest = json.loads((fresh_trial/'manifest.json').read_text())
    scope = manifest['htd_loaded_zero']
    assert scope['enabled'] and scope['duration_s'] == 10.
    assert scope['scope'] == 'loaded_zero_only'
    assert scope['policy_path'] == str(setup.PACKAGE/'policy.ts')
    assert scope['training_source'] == str(setup.PACKAGE/'runtime_env.yaml')
    assert scope['native_hold_source'] == str(setup.PROFILE/'carry_joint_targets.json')
    assert manifest['simulation_rate_limit'] == .5 and manifest['simulation_rate_start_phase_s'] == 90.
    assert manifest['htd_backlash_limit_timeconst_s'] == .005
    assert 'htd_initial_state' not in manifest
    robot = yaml.safe_load((fresh_trial/'robot.yaml').read_text())
    expected_robot = copy.deepcopy(original_robot)
    expected_robot['igris_c']['controller']['gain_parameter_path'] = str(fresh_trial/'htd_gc_params.yaml')
    assert robot == expected_robot
    params = yaml.safe_load((fresh_trial/'htd_gc_params.yaml').read_text())
    expected_params = copy.deepcopy(original_params)
    for key in ('gravity_comp_gain', 'gravity_comp_offset'):
        expected_params['igris_c_param_server']['ros__parameters'][key][:15] = [0.]*15
    assert params == expected_params
    assert evidence['environment']['IGRIS_ROOM_PROFILE_LIBRARY'] == str(fresh_trial/'htd_output_coordinates.so')
    assert scope['pacing_request_path'] == str(fresh_trial/'htd_pacing.request')
    assert not Path(scope['pacing_request_path']).exists()
    assert not Path(scope['pacing_ack_path']).exists()
    assert (fresh_trial/'htd_loaded_preflight.json').exists()
    with pytest.raises(FileExistsError, match='fresh'):
        setup.prepare(fresh_trial, enabled=True, environment={})


def test_current_grasp_setup_records_explicit_domain_and_actual_clearance_override(fresh_trial):
    manifest = json.loads((fresh_trial/'manifest.json').read_text())
    manifest['walking_transfer']['clearance_minimum_rise_m'] = .02
    (fresh_trial/'manifest.json').write_text(json.dumps(manifest))
    scene = (fresh_trial/'scene.xml').read_bytes()
    with pytest.raises(ValueError, match='current.grasp'):
        setup.prepare(fresh_trial, enabled=True, environment={})
    assert (fresh_trial/'scene.xml').read_bytes() == scene
    evidence = setup.prepare(fresh_trial, enabled=True, current_grasp_reference=True, environment={})
    scope = evidence['manifest_scope']
    assert scope['current_grasp_reference'] is True
    assert scope['clearance_minimum_rise_m'] == .02
    assert scope['runtime_domain_validated'] is False
    assert scope['checkpoint_sha256'] == setup.EXPECTED['model.pt']
    assert evidence['coordinator_contract']['clearance_minimum_rise_m'] == .02


def test_current_grasp_flag_requires_loaded_scope_even_when_setup_disabled(tmp_path):
    with pytest.raises(ValueError, match='scope'):
        setup.prepare(tmp_path/'unused', enabled=False, current_grasp_reference=True)
    assert not (tmp_path/'unused').exists()


def test_command_observer_requires_explicit_loaded_scope_before_input_reads(tmp_path):
    with pytest.raises(ValueError, match='observer.*loaded'):
        setup.prepare(tmp_path/'unused', command_observer=True)
    with pytest.raises(ValueError, match='observer.*loaded'):
        transport_trial.run(tmp_path/'unused', tmp_path/'absent', htd_command_observer=True)
    assert not (tmp_path/'unused').exists()


def test_wrist_mode_entry_requires_loaded_observed_current_grasp_context(tmp_path):
    with pytest.raises(ValueError,match='wrist.*entry'):
        setup.prepare(tmp_path/'unused',enabled=True,wrist_mode_entry=True)
    with pytest.raises(ValueError,match='wrist.*entry'):
        transport_trial.run(tmp_path/'unused',tmp_path/'absent',htd_wrist_mode_entry=True)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('wrist,goal,grace',[(False,False,False),(True,False,False),(True,True,False),(True,True,True)])
def test_command_observer_build_is_trial_local_unarmed_and_hash_pinned(fresh_trial,wrist,goal,grace):
    if goal:
        manifest=json.loads((fresh_trial/'manifest.json').read_text())
        manifest['walking_transfer']['clearance_lift_m']=.05
        manifest['walking_transfer']['clearance_minimum_rise_m']=.02
        (fresh_trial/'manifest.json').write_text(json.dumps(manifest))
    evidence=setup.prepare(fresh_trial,enabled=True,command_observer=True,
        current_grasp_reference=wrist,wrist_mode_entry=wrist,goal_hold=goal,contact_grace=grace,environment={})
    scope=evidence['manifest_scope'];observer=scope['command_observer']
    from htd_command_scope import validate
    environment=validate(fresh_trial,observer)
    assert environment['IGRIS_HTD_COMMAND_OBSERVER_EXE']=='/igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'
    assert scope['setup_sha256'][observer['library']]==observer['sha256']
    assert str(fresh_trial/'htd_command_observer.json') in scope['setup_sha256']
    assert 'LD_PRELOAD' not in evidence['environment']
    assert not Path(observer['arm_path']).exists()
    assert not Path(observer['trace_path']).exists()
    if wrist:
        from htd_wrist_entry import WristModeEntry
        assert scope['wrist_mode_entry']==WristModeEntry().contract
        assert str(setup.HERE/'htd_wrist_entry.py') in scope['setup_sha256']
    else:
        assert 'wrist_mode_entry' not in scope

    if goal:
        from htd_goal_hold import GoalHold
        assert scope['clearance_minimum_rise_m']==.02
        assert scope['scope']=='loaded_goal_hold_only'
        assert scope['goal_hold']==GoalHold().contract
        if not grace:assert scope['grasp_contact_rule']=='per-hand palm + tip'
        assert str(setup.HERE/'htd_goal_hold.py') in scope['setup_sha256']
        assert GoalHold().contract['controller_source'] in scope['setup_sha256']
    if grace:
        from htd_grasp_retention import GraspRetention
        assert scope['grasp_retention']==GraspRetention(.1,4.).contract
        assert str(setup.HERE/'htd_grasp_retention.py') in scope['setup_sha256']


def test_goal_hold_requires_explicit_complete_scope_before_input_reads(tmp_path):
    with pytest.raises(ValueError,match='Goal hold'):
        setup.prepare(tmp_path/'unused',goal_hold=True)
    with pytest.raises(ValueError,match='Goal hold'):
        transport_trial.run(tmp_path/'unused',tmp_path/'absent',htd_loaded_goal_hold=True)
    assert not (tmp_path/'unused').exists()


def test_contact_grace_requires_explicit_goal_scope_before_input_reads(tmp_path):
    with pytest.raises(ValueError,match='goal hold'):
        setup.prepare(tmp_path/'unused',contact_grace=True)
    with pytest.raises(ValueError,match='goal hold'):
        transport_trial.run(tmp_path/'unused',tmp_path/'absent',htd_contact_grace=True)
    assert not (tmp_path/'unused').exists()


def test_candidate_option_defaults_none_and_requires_scope_before_reads(tmp_path):
    assert inspect.signature(transport_trial.run).parameters['htd_policy_package'].default is None
    for flags in ({}, dict(enabled=True, current_grasp_reference=True)):
        with pytest.raises(ValueError, match='Candidate.*scope'):
            setup.prepare(tmp_path/'unused', policy_package=tmp_path/'absent', **flags)
    with pytest.raises(ValueError, match='Candidate.*scope'):
        transport_trial.run(tmp_path/'unused', tmp_path/'motion', htd_policy_package=tmp_path/'absent')
    assert not (tmp_path/'unused').exists()


def test_candidate_validation_precedes_trial_mutations_and_transport_preparation(fresh_trial, package, tmp_path, monkeypatch):
    directory, _ = package
    (directory/'model.pt').write_bytes(b'not the pinned checkpoint')
    before = {p.name: p.read_bytes() for p in fresh_trial.iterdir() if p.is_file()}
    def forbidden(*args, **kwargs): raise AssertionError('No preparation subprocess allowed')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='hash'):
        setup.prepare(fresh_trial, enabled=True, current_grasp_reference=True, command_observer=True,
                      wrist_mode_entry=True, goal_hold=True, policy_package=directory)
    assert before == {p.name: p.read_bytes() for p in fresh_trial.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='hash'):
        transport_trial.run(tmp_path/'unused', tmp_path/'motion', htd_loaded_zero=True,
            htd_current_grasp_reference=True, htd_command_observer=True, htd_wrist_mode_entry=True,
            htd_loaded_goal_hold=True, htd_policy_package=directory)
    assert not (tmp_path/'unused').exists()


def test_explicit_candidate_setup_records_package_and_continuation_not_frozen_history(fresh_trial, package):
    directory, descriptor = package
    evidence = setup.prepare(fresh_trial, enabled=True, current_grasp_reference=True,
        command_observer=True, wrist_mode_entry=True, goal_hold=True, policy_package=directory)
    scope = evidence['manifest_scope']
    assert scope['policy_selection']['selection'] == 'evaluated_continuation'
    assert scope['policy_path'] == str(directory/'policy.ts')
    assert scope['checkpoint_path'] == str(directory/'model.pt')
    assert scope['checkpoint_sha256'] == descriptor['artifact_sha256']['model.pt']
    assert scope['historically_trained_on'] == descriptor['historically_trained_on']
    assert scope['native_hold_source'] == str(setup.PROFILE/'carry_joint_targets.json')
    assert scope['runtime_domain_validated'] is False
    assert scope['runtime_domain'] == 'unvalidated actual acquired grasp; selected continuation weights; frozen profile76 owned15 contract'
    assert str(setup.HERE/'htd_candidate_package.py') in scope['setup_sha256']
    assert evidence['host_actor_warmed_history_unadvanced'] is True
