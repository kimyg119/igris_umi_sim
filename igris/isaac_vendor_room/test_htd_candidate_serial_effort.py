"""SIM candidate cap metadata cannot bypass frozen actuator/profile contracts."""
import copy
import json
from pathlib import Path

import pytest
import yaml

import htd_candidate_package as candidate
from test_htd_candidate_package import package, write_descriptor
from test_htd_posture_candidate import posture_package, repin

FIELD = 'native_serial_effort_profile'


def runtime_document(directory):
    return yaml.load((directory/'runtime_env.yaml').read_text(), Loader=yaml.BaseLoader)


def put_runtime(directory, document):
    (directory/'runtime_env.yaml').write_text(yaml.safe_dump(document))


@pytest.mark.parametrize('location', ['descriptor', 'deployment', 'runtime'])
@pytest.mark.parametrize('value', [None, {'schema': 'native_serial_effort_profile/v1', 'simulation_only': True}])
def test_incomplete_or_null_cap_declaration_cannot_be_silently_ignored(posture_package, location, value):
    # An unknown opt-in must not pass as the unchanged legacy actuator contract.
    directory, descriptor = posture_package
    if location == 'descriptor':
        descriptor[FIELD] = value
    elif location == 'deployment':
        spec = json.loads((directory/'deployment.json').read_text())
        spec[FIELD] = value
        (directory/'deployment.json').write_text(json.dumps(spec))
    else:
        runtime = runtime_document(directory)
        runtime[FIELD] = value
        put_runtime(directory, runtime)
    repin(directory, descriptor)
    with pytest.raises(ValueError, match='serial|effort|profile'):
        candidate.resolve_package(directory)


def typed_baseline_actuators():
    node = yaml.compose((candidate.PACKAGE/'runtime_env.yaml').read_text(), Loader=yaml.SafeLoader)
    for key in ('robot_cfg', 'actuators'):
        node = next(value for name, value in node.value if name.value == key)
    return yaml.safe_load(yaml.serialize(node))


@pytest.fixture
def capped_posture_package(posture_package):
    from wbc_base.serial_effort_profile import profile_for_actuators, configured_actuators, NATIVE_PARAMS
    directory, descriptor = posture_package
    original = typed_baseline_actuators()
    metadata = profile_for_actuators(original)
    runtime = runtime_document(directory)
    runtime['robot_cfg']['actuators'] = configured_actuators(original, metadata)
    runtime[FIELD] = metadata
    put_runtime(directory, runtime)
    spec = json.loads((directory/'deployment.json').read_text())
    spec[FIELD] = metadata
    (directory/'deployment.json').write_text(json.dumps(spec))
    descriptor[FIELD] = metadata
    helper = candidate.ROOT/'igris/igris-c-wbc-base/src/wbc_base/serial_effort_profile.py'
    descriptor['provenance'].update({str(path): candidate.sha(path) for path in (helper, NATIVE_PARAMS)})
    repin(directory, descriptor)
    return directory, descriptor


def test_exact_owned15_caps_roundtrip_and_real_native_adapter_preserves_delivered_limits(capped_posture_package):
    import numpy as np
    from htd_probe_config import resolve_artifacts
    from htd_serial_pd import SerialPD
    from htd_native_probe import TargetLimits, load_gains
    directory, descriptor = capped_posture_package
    selection = candidate.resolve_package(directory)
    assert selection['selection'] == 'evaluated_posture_continuation'
    metadata = selection[FIELD]
    changed = {
        'Joint_Waist_Roll': 60., 'Joint_Waist_Pitch': 60.,
        'Joint_Ankle_Pitch_Left': 90., 'Joint_Ankle_Roll_Left': 90.,
        'Joint_Ankle_Pitch_Right': 90., 'Joint_Ankle_Roll_Right': 90.,
    }
    assert set(metadata['changed_joint_names']) == set(changed)
    for key in ('configured_effort_limits_Nm', 'configured_effort_limits_sim_Nm'):
        assert {name: metadata[key][name] for name in changed} == changed
        assert metadata[key]['Joint_Hip_Pitch_Left'] == 150.
        assert metadata[key]['Joint_Shoulder_Pitch_Left'] == 60.
    scope = dict(policy_selection=selection, current_grasp_reference=True, scope='loaded_goal_hold_only',
                 goal_hold={}, command_observer={}, wrist_mode_entry={},
                 historically_trained_on=selection['historically_trained_on'])
    assert candidate.validate_runtime_selection(scope) == selection
    artifacts = resolve_artifacts(policy_path=directory/'policy.ts', deployment_path=directory/'deployment.json',
        training_source=directory/'runtime_env.yaml', native_hold_source=directory/'profile/carry_joint_targets.json')
    kp, kd, _ = load_gains()
    old = SerialPD(artifacts.hold_q, kp, kd, TargetLimits(), delay_steps=[0, 1, 2, 2],
                   training_source=candidate.PACKAGE/'runtime_env.yaml')
    new = SerialPD(artifacts.hold_q, kp, kd, TargetLimits(), delay_steps=[0, 1, 2, 2],
                   training_source=directory/'runtime_env.yaml')
    q = artifacts.hold_q.copy()
    for tick in range(4):
        target = q.copy();target[:15] += (-1 if tick % 2 else 1)*9.
        state = dict(q_output=q.copy(), dq_output=np.zeros(31), q_encoder=q.copy(),
                     dq_encoder=np.zeros(31), sim=tick*.005, sequence=2*(tick+1), monotonic=10.)
        before = old.step(target, state, native_time=tick*.005, wall_time=10.)
        after = new.step(target, state, native_time=tick*.005, wall_time=10.)
        np.testing.assert_array_equal(after.tau, before.tau)
        np.testing.assert_array_equal(after.delayed_target, before.delayed_target)
        np.testing.assert_array_equal(after.kp, before.kp)
        np.testing.assert_array_equal(after.kd, before.kd)
        np.testing.assert_array_equal(after.q, before.q)
        assert not after.native_clipped.any()
    assert old.contract['training_effort'][1] == 217.0253
    assert new.contract['training_effort'][1] == 60.
    assert old.contract['training_effort'][8] == 148.7251
    assert new.contract['training_effort'][8] == 90.
    assert new.contract['pd_dt_s'] == .005 and new.contract['max_output_native_lag_s'] == .002


@pytest.mark.parametrize('change', ['gain', 'delay', 'delay_bool_alias', 'delay_float_alias', 'class',
    'held_cap', 'owned_cap', 'sim_cap', 'joint_pattern', 'extra_limit_pattern', 'armature', 'extra_field'])
def test_cap_opt_in_cannot_hide_any_other_actuator_change(capped_posture_package, change):
    directory, descriptor = capped_posture_package
    runtime = yaml.safe_load((directory/'runtime_env.yaml').read_text())
    groups = runtime['robot_cfg']['actuators']
    if change == 'gain':groups['X8-120']['stiffness'] += 1.
    elif change == 'delay':groups['X8-120']['max_delay'] = 1
    elif change == 'delay_bool_alias':groups['X8-120']['min_delay'] = False
    elif change == 'delay_float_alias':groups['X8-120']['max_delay'] = 2.0
    elif change == 'class':groups['X8-120']['class_type'] = 'other:Actuator'
    elif change == 'held_cap':groups['X6-60']['effort_limit'] = 59.
    elif change == 'owned_cap':groups['PLA']['effort_limit']['Joint_Waist_Roll'] = 61.
    elif change == 'sim_cap':groups['PLA']['effort_limit_sim']['Joint_Ankle_Roll_Left'] = 89.
    elif change == 'joint_pattern':groups['X6-60']['joint_names_expr'].append('Unowned_joint')
    elif change == 'extra_limit_pattern':groups['PLA']['effort_limit']['Unowned_joint'] = 60.
    elif change == 'armature':groups['X8-120']['armature'] += .01
    elif change == 'extra_field':groups['X8-120']['unreviewed'] = True
    put_runtime(directory, runtime);repin(directory, descriptor)
    with pytest.raises(ValueError, match='actuator|Actuator|effort|Effort|contract|profile'):
        candidate.resolve_package(directory)


@pytest.mark.parametrize('change', ['native_hash', 'native_source', 'native_cap', 'configured_cap',
    'original_cap', 'sim_only_alias', 'joint_order', 'extra_key', 'schema', 'nan'])
def test_consistently_repinning_three_declarations_cannot_rewrite_audited_profile(capped_posture_package, change):
    directory, descriptor = capped_posture_package
    metadata = copy.deepcopy(descriptor[FIELD])
    if change == 'native_hash':metadata['native_params_sha256'] = '0'*64
    elif change == 'native_source':metadata['native_params_source'] = str(directory/'policy.ts')
    elif change == 'native_cap':metadata['native_serial_caps_Nm']['Joint_Waist_Roll'] = 61.
    elif change == 'configured_cap':metadata['configured_effort_limits_Nm']['Joint_Waist_Roll'] = 61.
    elif change == 'original_cap':metadata['original_effort_limits_Nm']['Joint_Waist_Roll'] = 60.
    elif change == 'sim_only_alias':metadata['simulation_only'] = 1
    elif change == 'joint_order':metadata['policy_joint_names'].reverse()
    elif change == 'extra_key':metadata['extra'] = True
    elif change == 'schema':metadata['schema'] = 'native_serial_effort_profile/v2'
    elif change == 'nan':metadata['configured_effort_limits_Nm']['Joint_Waist_Roll'] = float('nan')
    descriptor[FIELD] = metadata
    spec = json.loads((directory/'deployment.json').read_text());spec[FIELD] = metadata
    (directory/'deployment.json').write_text(json.dumps(spec))
    runtime = yaml.safe_load((directory/'runtime_env.yaml').read_text());runtime[FIELD] = metadata
    put_runtime(directory, runtime);repin(directory, descriptor)
    with pytest.raises(ValueError, match='serial|effort|profile'):
        candidate.resolve_package(directory)


@pytest.mark.parametrize('location', ['descriptor', 'deployment', 'runtime'])
def test_one_declaration_mismatch_rejected_after_all_artifacts_rehashed(capped_posture_package, location):
    directory, descriptor = capped_posture_package
    if location == 'descriptor':descriptor[FIELD]['simulation_only'] = False
    elif location == 'deployment':
        spec = json.loads((directory/'deployment.json').read_text());spec[FIELD]['simulation_only'] = False
        (directory/'deployment.json').write_text(json.dumps(spec))
    else:
        runtime = yaml.safe_load((directory/'runtime_env.yaml').read_text());runtime[FIELD]['simulation_only'] = False
        put_runtime(directory, runtime)
    repin(directory, descriptor)
    with pytest.raises(ValueError, match='declarations differ'):
        candidate.resolve_package(directory)


@pytest.mark.parametrize('case', ['undeclared_caps', 'declared_without_caps'])
def test_declaration_and_actual_runtime_limits_are_both_required(capped_posture_package, case):
    directory, descriptor = capped_posture_package
    runtime = yaml.safe_load((directory/'runtime_env.yaml').read_text())
    if case == 'undeclared_caps':
        descriptor.pop(FIELD);runtime.pop(FIELD)
        spec = json.loads((directory/'deployment.json').read_text());spec.pop(FIELD)
        (directory/'deployment.json').write_text(json.dumps(spec))
    else:runtime['robot_cfg']['actuators'] = typed_baseline_actuators()
    put_runtime(directory, runtime);repin(directory, descriptor)
    with pytest.raises(ValueError, match='actuator|Actuator'):
        candidate.resolve_package(directory)


def test_legacy_candidate_cannot_select_cap_profile(package):
    from wbc_base.serial_effort_profile import profile_for_actuators, configured_actuators
    directory, descriptor = package
    metadata = profile_for_actuators(typed_baseline_actuators())
    descriptor[FIELD] = metadata
    spec = json.loads((directory/'deployment.json').read_text());spec[FIELD] = metadata
    (directory/'deployment.json').write_text(json.dumps(spec))
    runtime = runtime_document(directory);runtime[FIELD] = metadata
    runtime['robot_cfg']['actuators'] = configured_actuators(typed_baseline_actuators(), metadata)
    put_runtime(directory, runtime);repin(directory, descriptor)
    with pytest.raises(ValueError, match='V2'):
        candidate.resolve_package(directory)


@pytest.mark.parametrize('missing', ['helper', 'native_params'])
def test_selected_cap_profile_requires_current_source_and_native_parameter_evidence(capped_posture_package, missing):
    directory, descriptor = capped_posture_package
    path = (candidate.ROOT/'igris/igris-c-wbc-base/src/wbc_base/serial_effort_profile.py'
            if missing == 'helper' else Path(descriptor[FIELD]['native_params_source']))
    descriptor['provenance'].pop(str(path))
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError, match='provenance|evidence'):
        candidate.resolve_package(directory)
