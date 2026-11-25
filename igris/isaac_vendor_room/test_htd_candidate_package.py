"""Read-only package selection tests; fixture evidence is NOT a policy evaluation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile

import pytest

import htd_loaded_setup as setup
import htd_candidate_package as candidate


def write_descriptor(directory, descriptor):
    (directory/'loaded_candidate.json').write_text(json.dumps(descriptor, indent=2)+'\n')


@pytest.fixture
def package():
    with tempfile.TemporaryDirectory(prefix='htd_candidate_test_', dir=setup.ROOT/'outputs') as temporary:
        directory = Path(temporary)
        for name in setup.EXPECTED:
            shutil.copyfile(setup.PACKAGE/name, directory/name)
        for name in ('provenance.json', 'evaluation.json'):
            (directory/name).write_text('{"scope":"offline test fixture; no evaluation claim"}\n')
        descriptor = dict(schema='htd_loaded_candidate/v1',
            artifact_sha256={name: setup.sha(directory/name) for name in setup.EXPECTED},
            profile_directory=str(setup.PROFILE),
            profile_sha256=json.loads((setup.PACKAGE/'deployment.json').read_text())['carry_profile']['profile_sha256'],
            historically_trained_on='TEST ONLY: continuation fixture on unchanged profile76',
            provenance={str(directory/'provenance.json'): setup.sha(directory/'provenance.json')},
            evaluation_evidence={str(directory/'evaluation.json'): setup.sha(directory/'evaluation.json')})
        write_descriptor(directory, descriptor)
        yield directory, descriptor


def test_default_remains_frozen_and_requires_no_descriptor():
    selection = candidate.resolve_package()
    assert selection['selection'] == 'frozen1750_profile76'
    assert selection['package_directory'] == str(setup.PACKAGE)
    assert selection['artifact_sha256'] == setup.EXPECTED
    assert 'descriptor_path' not in selection
    assert selection['historically_trained_on'] == 'originalV2 profile68; untrained-on76 runtime correction'


def test_explicit_package_preserves_real_actor_defaults_and_records_selection(package):
    directory, descriptor = package
    selection = candidate.resolve_package(directory)
    assert selection['selection'] == 'evaluated_continuation'
    assert selection['descriptor_sha256'] == setup.sha(directory/'loaded_candidate.json')
    assert selection['profile_sha256'] == descriptor['profile_sha256']
    assert selection['historically_trained_on'] == descriptor['historically_trained_on']
    assert selection['evaluation_evidence'] == descriptor['evaluation_evidence']


@pytest.mark.parametrize('name', [*setup.EXPECTED, 'provenance.json', 'evaluation.json'])
def test_each_pinned_artifact_or_evidence_tamper_rejected(package, name):
    directory, _ = package
    with (directory/name).open('ab') as stream: stream.write(b'\ntamper')
    with pytest.raises(ValueError, match='hash'):
        candidate.resolve_package(directory)


@pytest.mark.parametrize('change', ['schema', 'missing_artifact', 'extra_artifact', 'bad_hash',
    'profile_hash', 'profile_directory', 'no_provenance', 'no_evaluation', 'no_history'])
def test_descriptor_fail_closed(package, change):
    directory, descriptor = package
    if change == 'schema': descriptor['schema'] = 'unknown/v1'
    if change == 'missing_artifact': descriptor['artifact_sha256'].pop('model.pt')
    if change == 'extra_artifact': descriptor['artifact_sha256']['other.ts'] = '0'*64
    if change == 'bad_hash': descriptor['artifact_sha256']['model.pt'] = 'not-a-hash'
    if change == 'profile_hash': descriptor['profile_sha256']['report.json'] = '0'*64
    if change == 'profile_directory': descriptor['profile_directory'] = str(directory)
    if change == 'no_provenance': descriptor['provenance'] = {}
    if change == 'no_evaluation': descriptor['evaluation_evidence'] = {}
    if change == 'no_history': descriptor['historically_trained_on'] = ' '
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError): candidate.resolve_package(directory)


def test_explicit_default_path_is_not_an_implicit_opt_in():
    with pytest.raises((ValueError, FileNotFoundError), match='descriptor|loaded_candidate'):
        candidate.resolve_package(setup.PACKAGE)


def test_outside_project_and_symlink_escapes_rejected_before_reads(package, tmp_path):
    directory, descriptor = package
    with pytest.raises(ValueError, match='project'): candidate.resolve_package(tmp_path)
    external = tmp_path/'evidence.json'; external.write_text('{}')
    descriptor['evaluation_evidence'] = {str(external): setup.sha(external)}
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError, match='project'): candidate.resolve_package(directory)
    descriptor['evaluation_evidence'] = {str(directory/'evaluation.json'): setup.sha(external)}
    (directory/'evaluation.json').unlink(); (directory/'evaluation.json').symlink_to(external)
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError, match='project'): candidate.resolve_package(directory)


def test_artifact_symlink_cannot_replace_exact_package_paths(package):
    directory, _ = package
    (directory/'policy.ts').unlink(); (directory/'policy.ts').symlink_to(setup.PACKAGE/'policy.ts')
    with pytest.raises(ValueError, match='symlink|package path'): candidate.resolve_package(directory)


@pytest.mark.parametrize('change', ['offset', 'profile', 'gain', 'delay'])
def test_repinning_files_cannot_change_profile_offsets_gains_or_delay(package, change):
    directory, descriptor = package
    spec = json.loads((directory/'deployment.json').read_text())
    if change == 'offset': spec['default_joint_pos'][2] += .01
    if change == 'profile': spec['carry_profile']['policy_joint_offsets']['Joint_Waist_Yaw'] += .01
    if change in ('offset', 'profile'):
        (directory/'deployment.json').write_text(json.dumps(spec))
    else:
        path = directory/'runtime_env.yaml'; original = path.read_text()
        changed = original.replace('122.136354', '123.136354', 1) if change == 'gain' else original.replace('max_delay: 2', 'max_delay: 1', 1)
        assert changed != original
        path.write_text(changed)
    descriptor['artifact_sha256'] = {name: setup.sha(directory/name) for name in setup.EXPECTED}
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError, match='contract|profile|actuator|default|offset'):
        candidate.resolve_package(directory)


def test_training_provenance_reset_metadata_may_change_without_changing_runtime(package):
    directory, descriptor = package
    path = directory/'runtime_env.yaml'
    path.write_text(path.read_text()+'\ncaptured_stance_training: true\n')
    descriptor['artifact_sha256']['runtime_env.yaml'] = setup.sha(path)
    write_descriptor(directory, descriptor)
    assert candidate.resolve_package(directory)['artifact_sha256']['runtime_env.yaml'] == setup.sha(path)


def test_runtime_revalidates_descriptor_hash_and_scope(package):
    directory, descriptor = package
    selection = candidate.resolve_package(directory)
    scope = dict(policy_selection=selection, current_grasp_reference=True, scope='loaded_goal_hold_only',
        goal_hold={}, command_observer={}, wrist_mode_entry={}, historically_trained_on=selection['historically_trained_on'])
    assert candidate.validate_runtime_selection(scope) == selection
    for key in ('current_grasp_reference', 'goal_hold', 'command_observer', 'wrist_mode_entry'):
        bad = copy.deepcopy(scope); bad.pop(key)
        with pytest.raises(ValueError, match='scope'): candidate.validate_runtime_selection(bad)
    transport=dict(scope,scope='loaded_transport',duration_s=180.,
                   transport={'schema':'htd_loaded_transport/v1'},grasp_retention={})
    assert candidate.validate_runtime_selection(transport)==selection
    for key in ('duration_s','transport','grasp_retention'):
        bad=copy.deepcopy(transport);bad.pop(key)
        with pytest.raises(ValueError,match='scope'):candidate.validate_runtime_selection(bad)
    descriptor['historically_trained_on'] += '; revised'
    write_descriptor(directory, descriptor)
    with pytest.raises(ValueError, match='selection|descriptor'): candidate.validate_runtime_selection(scope)


def test_runtime_missing_selection_stays_default_and_unknown_selection_rejected():
    assert candidate.validate_runtime_selection({}) == candidate.resolve_package()
    with pytest.raises(ValueError, match='selection'):
        candidate.validate_runtime_selection({'policy_selection': {'selection': 'automatic'}})
