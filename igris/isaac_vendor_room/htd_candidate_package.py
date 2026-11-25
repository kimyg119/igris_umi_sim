"""Read-only opt-in routing for evaluated loaded SIM continuations.

``loaded_candidate.json`` schema ``htd_loaded_candidate/v1`` requires:
* artifact_sha256: exact model.pt/policy.ts/deployment.json/runtime_env.yaml hashes;
* profile_directory and profile_sha256: unchanged frozen profile76 identity;
* historically_trained_on: explicit continuation provenance, not inherited1750 text;
* provenance and evaluation_evidence: nonempty project path -> SHA256 mappings.

V2 permits audited held-arm/distal-mass profiles while preserving owned15
control. A proxy filter requires the exact ``fixed_wrist_contact_filter``
declaration in the descriptor, profile report and deployment metadata.
An optional native serial effort profile requires matching typed metadata in
the V2 descriptor, deployment and actual runtime YAML, independently checked
against the frozen actuator configuration and native parameter file.

Evidence paths may be absolute or project-relative. Hash validation records the
selected evidence; it does not certify its conclusions or grant hardware use.
No descriptor is read for the default frozen1750 selection. This module never
prepares a scene, starts physics, discovers services, or publishes commands.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
P = ROOT/'outputs/hlik_transport_20260918_2005'
PACKAGE = P/'htd_carry_profile76_runtime1750_candidate02'
PROFILE = P/'htd_carry_profile76_02'
EXPECTED = {
    'model.pt': 'a3f3bd284fa5945471b17e24486dc87e7ccf408d3814290dd54e7fab81df445d',
    'policy.ts': '1406df99a009e35f2ef85fa95d309ed7d2a7270b73e6a9c9e1f01eeed85fafdd',
    'deployment.json': 'c193c2f202ce05d5b7756cea2188632b749d11720663d9a23bb1b39923271e4b',
    'runtime_env.yaml': 'd3e909108c74cf78725875b8b69a06c66c118db404686d7aaf5dff8f79b2e244',
}
DEFAULT_HISTORY = 'originalV2 profile68; untrained-on76 runtime correction'
SCHEMA = 'htd_loaded_candidate/v1'
POSTURE_SCHEMA = 'htd_loaded_candidate/v2'
SERIAL_EFFORT_FIELD = 'native_serial_effort_profile'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project_path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError('Candidate requires a project path')
    path = Path(value)
    if not path.is_absolute(): path = ROOT/path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Candidate paths must remain inside this project')
    return path


def _verify(path, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Invalid candidate SHA256 hash: '+str(path))
    if sha(path) != expected:
        raise ValueError('Loaded package/evidence hash changed: '+str(path))


def require_candidate_scope(package, *, enabled, current_grasp_reference, command_observer,
                            wrist_mode_entry, goal_hold):
    if package is not None and not all(value is True for value in
            (enabled, current_grasp_reference, command_observer, wrist_mode_entry, goal_hold)):
        raise ValueError('Candidate policy package requires explicit loaded current-grasp goal-hold scope with observed wrist entry')


def _evidence_paths(mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Candidate requires nonempty '+label+' path/hash evidence')
    result = {}
    for name, expected in mapping.items():
        path = _project_path(name)
        _verify(path, expected)
        if str(path) in result:
            raise ValueError('Duplicate candidate evidence path: '+str(path))
        result[str(path)] = expected
    return result


def _safe_runtime_field(directory, *keys):
    """Construct only the requested plain subtree, never Python-tagged config."""
    try:
        node = yaml.compose((directory/'runtime_env.yaml').read_text(), Loader=yaml.SafeLoader)
        for key in keys:
            if not isinstance(node, yaml.MappingNode):
                raise ValueError('Serial effort profile requires a runtime mapping')
            matches = [value for name, value in node.value
                       if isinstance(name, yaml.ScalarNode) and name.value == key]
            if not matches:
                return False, None
            if len(matches) != 1:
                raise ValueError('Duplicate runtime serial effort profile field: '+key)
            node = matches[0]
        return True, yaml.safe_load(yaml.serialize(node))
    except yaml.YAMLError as error:
        raise ValueError('Invalid runtime serial effort profile field') from error


def _serial_effort_declaration(directory, descriptor, spec, *, posture):
    present, runtime = _safe_runtime_field(directory, SERIAL_EFFORT_FIELD)
    selected = (SERIAL_EFFORT_FIELD in descriptor, SERIAL_EFFORT_FIELD in spec, present)
    if not any(selected):
        return None
    if not posture or not all(selected):
        raise ValueError('Serial effort profile requires explicit V2 descriptor/deployment/runtime declarations')
    declarations = [descriptor[SERIAL_EFFORT_FIELD], spec[SERIAL_EFFORT_FIELD], runtime]
    if any(not isinstance(value, dict) or not value for value in declarations):
        raise ValueError('Serial effort profile declarations must be nonempty mappings')
    try:
        encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in declarations]
    except (TypeError, ValueError) as error:
        raise ValueError('Serial effort profile requires finite typed JSON metadata') from error
    if len(set(encoded)) != 1:
        raise ValueError('Serial effort profile descriptor/deployment/runtime declarations differ')
    return declarations[0]


def _validate_posture_profile(profile, spec, frozen):
    """Recompute metadata from pinned audit files; only declared posture/mass differ."""
    source_path=str(ROOT/'igris/igris-c-wbc-base/src')
    if source_path not in sys.path:sys.path.insert(0,source_path)
    from wbc_base.carry_profile import configure_carry_profile
    original=frozen['carry_profile']
    report=json.loads((profile/'report.json').read_text())
    old_report=json.loads((PROFILE/'report.json').read_text())
    for mapping in ('original_asset_layer_hashes','verified_provenance_sha256'):
        for path in report[mapping]:_project_path(path)
    for key in ('source_asset','proxy_asset','source_checkpoint'):_project_path(report[key])
    for key in ('source_asset','source_asset_sha256','original_asset_layer_hashes',
                'mass_property_mode','numerical_contract'):
        if report.get(key)!=old_report.get(key):
            raise ValueError('Posture profile changes frozen source/numerical contract: '+key)
    old_changes={x['body_path']:x for x in old_report['mass_property_changes']}
    for item in report['mass_property_changes']:
        old=old_changes.get(item['body_path'])
        if old is None or any(item.get(k)!=old.get(k)for k in ('original_mass_kg','native_bare_mass_kg')):
            raise ValueError('Posture profile changes native bare mass or distal body identity')
    cfg=SimpleNamespace(phase_gait=True,observation_space={'policy':120,'critic':131},action_space=15,
        robot_cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=report['source_asset']),
            init_state=SimpleNamespace(joint_pos={**original['baseline_policy_joint_offsets'],
                                                  **original['held_joint_targets_rad']})))
    computed=configure_carry_profile(cfg,profile)
    for key in ('schema','held_joint_names','policy_joint_offsets','baseline_policy_joint_offsets',
                'numerical_contract','reset_noise_scope','soft_limit_reward_scope'):
        if computed.get(key)!=original.get(key):
            raise ValueError('Posture profile changes frozen owned15 contract: '+key)
    if spec.get('carry_profile')!=computed:
        raise ValueError('Posture deployment differs from audited profile metadata')
    _validate_proxy_opinions(profile,computed,old_report)
    return computed


def _validate_proxy_opinions(profile,metadata,old_report):
    """Normalize eight mass values and an explicitly reviewed pair relation."""
    from pxr import Sdf
    from scipy.spatial.transform import Rotation
    baseline=Sdf.Layer.CreateAnonymous();changed=Sdf.Layer.CreateAnonymous()
    if (not baseline.ImportFromString((PROFILE/'carry_load_proxy.usda').read_text())
            or not changed.ImportFromString((profile/'carry_load_proxy.usda').read_text())):
        raise ValueError('Cannot parse posture proxy layer')
    contact_filter=metadata.get('fixed_wrist_contact_filter')
    if contact_filter is not None:
        from wbc_base.carry_profile import validate_fixed_wrist_contact_filter
        validate_fixed_wrist_contact_filter(dict(fixed_wrist_contact_filter=contact_filter,
            proxy_asset=str(profile/'carry_load_proxy.usda')))
        path=contact_filter['body_path']
        prim=changed.GetPrimAtPath(path)
        relation=changed.GetRelationshipAtPath(path+'.'+contact_filter['relationship'])
        if (relation.custom or relation.variability!=Sdf.VariabilityUniform
                or set(relation.ListInfoKeys())!={'custom','variability','targetPaths'}):
            raise ValueError('Posture filter relationship adds unreviewed metadata')
        prim.RemoveProperty(relation)
        prim.SetInfo('apiSchemas',baseline.GetPrimAtPath(path).GetInfo('apiSchemas'))
    for body in old_report['mass_property_changes']:
        path=body['body_path'];values={}
        for name in ('mass','centerOfMass','diagonalInertia','principalAxes'):
            attribute=path+'.physics:'+name
            before=baseline.GetAttributeAtPath(attribute);after=changed.GetAttributeAtPath(attribute)
            if before is None or after is None:raise ValueError('Missing distal mass attribute')
            values[name]=after.default
            after.default=before.default
        expected=metadata['expected_distal_properties'][Path(path).name]
        quat=values['principalAxes']
        rotation=Rotation.from_quat([quat.GetReal(),*quat.GetImaginary()],scalar_first=True).as_matrix()
        inertia=rotation@np.diag(values['diagonalInertia'])@rotation.T
        if (not np.isfinite(values['mass']) or abs(values['mass']-expected['mass_kg'])>3e-7
                or not np.allclose(values['centerOfMass'],expected['com_body_m'],atol=1e-7,rtol=0)
                or not np.allclose(inertia,expected['inertia_at_com_body_kg_m2'],atol=1e-7,rtol=0)):
            raise ValueError('Posture proxy serialized mass/COM/inertia differs from audit')
    if changed.ExportToString()!=baseline.ExportToString():
        raise ValueError('Posture proxy changes fields beyond two distal inertias and explicitly reviewed filter')


def _validate_control(directory, spec, frozen, profile=PROFILE, *, posture=False,
                      serial_effort_profile=None):
    # Export metadata can describe new training/evaluation; the deployed numerical
    # interface, immutable profile geometry, offsets and runtime PD stay pinned.
    for key in ('schema', 'scope', 'format', 'contract', 'input_shape', 'output_shape',
                'policy_joint_names', 'observation_terms', 'phase_gait', 'phase_period_s',
                'phase_protocol', 'history_layout', 'all_joint_names', 'default_joint_pos',
                'world_xy_controller'):
        if posture and key in ('contract','default_joint_pos'):continue
        if spec.get(key) != frozen.get(key):
            raise ValueError('Candidate changes frozen deployment contract: '+key)
    if posture:
        declared=_validate_posture_profile(profile,spec,frozen)
        expected_contract={**frozen['contract'],'action_offset_profile_sha256':declared['profile_sha256']}
        if spec.get('contract')!=expected_contract:
            raise ValueError('Posture candidate changes frozen deployment contract')
        expected=list(frozen['default_joint_pos'])
        for name,value in declared['held_joint_targets_rad'].items():
            expected[frozen['all_joint_names'].index(name)]=float(np.float32(value))
        if spec.get('default_joint_pos')!=expected:
            raise ValueError('Posture candidate changes owned15/default joint contract')
    for key in ('schema', 'directory', 'profile_sha256', 'held_joint_names', 'held_joint_targets_rad',
                'policy_joint_offsets', 'baseline_policy_joint_offsets', 'numerical_contract',
                'expected_distal_properties', 'expected_distal_mass_kg', 'added_payload_mass_kg',
                'reset_noise_scope', 'soft_limit_reward_scope'):
        if not posture and spec.get('carry_profile', {}).get(key) != frozen['carry_profile'].get(key):
            raise ValueError('Candidate changes frozen profile76 contract: '+key)
    training = yaml.load((directory/'runtime_env.yaml').read_text(), Loader=yaml.BaseLoader)
    baseline = yaml.load((PACKAGE/'runtime_env.yaml').read_text(), Loader=yaml.BaseLoader)
    if serial_effort_profile is not None:
        source_path = str(ROOT/'igris/igris-c-wbc-base/src')
        if source_path not in sys.path:sys.path.insert(0, source_path)
        from wbc_base.serial_effort_profile import validate_profile
        _, original_actuators = _safe_runtime_field(PACKAGE, 'robot_cfg', 'actuators')
        present, configured_actuators = _safe_runtime_field(directory, 'robot_cfg', 'actuators')
        if not present or not isinstance(configured_actuators, dict):
            raise ValueError('Serial effort profile requires actual runtime actuators')
        # The pure helper recomputes the exact owned15 min(original,native)
        # limits and compares every other actuator field with this frozen input.
        validate_profile(serial_effort_profile, original_actuators, configured_actuators)
    elif training['robot_cfg']['actuators'] != baseline['robot_cfg']['actuators']:
        raise ValueError('Candidate changes frozen runtime actuator contract')
    if not posture and training['robot_cfg']['init_state']['joint_pos'] != baseline['robot_cfg']['init_state']['joint_pos']:
        raise ValueError('Candidate changes frozen runtime joint defaults')
    if posture:
        positions=training['robot_cfg']['init_state']['joint_pos']
        if set(positions)!=set(baseline['robot_cfg']['init_state']['joint_pos']):
            raise ValueError('Posture runtime changes joint-default patterns')
        held=set(spec['carry_profile']['held_joint_names'])
        if any(positions[n]!=v for n,v in baseline['robot_cfg']['init_state']['joint_pos'].items()if n not in held):
            raise ValueError('Posture runtime changes owned15 joint defaults')
    if training['sim']['dt'] != baseline['sim']['dt'] or training['decimation'] != baseline['decimation']:
        raise ValueError('Candidate changes frozen runtime timestep contract')
    from htd_probe_config import resolve_artifacts
    from htd_policy import HTDPolicy
    artifacts = resolve_artifacts(policy_path=directory/'policy.ts', deployment_path=directory/'deployment.json',
        training_source=directory/'runtime_env.yaml', native_hold_source=profile/'carry_joint_targets.json')
    if posture:
        from htd_native_probe import TargetLimits
        TargetLimits().check(artifacts.hold_q)
    # Callable120->15 boundary and finite CPU inference are checked before a
    # transport caller can create the output scene; no action history is consumed.
    HTDPolicy(artifacts.policy_path, artifacts.deployment_path)


def resolve_package(package=None):
    """Validate all inputs and return JSON-ready immutable-by-convention evidence."""
    for name, expected in EXPECTED.items(): _verify(PACKAGE/name, expected)
    frozen = json.loads((PACKAGE/'deployment.json').read_text())
    profile_hashes = frozen['carry_profile']['profile_sha256']
    for name, expected in profile_hashes.items():
        if Path(name).name != name: raise ValueError('Invalid frozen profile path')
        _verify(PROFILE/name, expected)
    if package is None:
        return dict(selection='frozen1750_profile76', package_directory=str(PACKAGE),
            artifact_sha256=EXPECTED.copy(), profile_directory=str(PROFILE),
            profile_sha256=profile_hashes, historically_trained_on=DEFAULT_HISTORY)
    directory = _project_path(package)
    descriptor_path = directory/'loaded_candidate.json'
    for name in (*EXPECTED, 'loaded_candidate.json'):
        if (directory/name).resolve() != directory/name:
            raise ValueError('Candidate artifact symlink changes exact package path: '+name)
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    if not isinstance(descriptor, dict) or descriptor.get('schema') not in (SCHEMA,POSTURE_SCHEMA):
        raise ValueError('Unsupported loaded candidate descriptor schema')
    posture=descriptor['schema']==POSTURE_SCHEMA
    hashes = descriptor.get('artifact_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(EXPECTED):
        raise ValueError('Candidate must pin exactly four artifact hashes')
    for name, expected in hashes.items(): _verify(directory/name, expected)
    profile=PROFILE
    if posture:
        profile=_project_path(descriptor.get('profile_directory'))
        profile_hashes=descriptor.get('profile_sha256')
        if (str(profile)!=descriptor.get('profile_directory') or not isinstance(profile_hashes,dict)
                or set(profile_hashes)!=set(frozen['carry_profile']['profile_sha256'])):
            raise ValueError('Posture candidate requires exact project profile directory and three hashes')
        for name,expected in profile_hashes.items():
            if (profile/name).resolve()!=profile/name:
                raise ValueError('Posture profile artifact symlink changes exact path')
            _verify(profile/name,expected)
    elif (descriptor.get('profile_directory') != str(PROFILE)
            or descriptor.get('profile_sha256') != profile_hashes):
        raise ValueError('Candidate must retain frozen profile76 directory and hashes')
    history = descriptor.get('historically_trained_on')
    if not isinstance(history, str) or not history.strip() or history == DEFAULT_HISTORY:
        raise ValueError('Candidate requires explicit continuation training history')
    provenance = _evidence_paths(descriptor.get('provenance'), 'provenance')
    evaluations = _evidence_paths(descriptor.get('evaluation_evidence'), 'evaluation')
    spec = json.loads((directory/'deployment.json').read_text())
    serial_effort_profile = _serial_effort_declaration(directory, descriptor, spec, posture=posture)
    if serial_effort_profile is not None:
        for path in (ROOT/'igris/igris-c-wbc-base/src/wbc_base/serial_effort_profile.py',
                     ROOT/'igris/configs/controller/params.yaml'):
            # _evidence_paths has already verified these exact current bytes.
            if str(path.resolve()) not in provenance:
                raise ValueError('Serial effort profile requires source/native-parameter provenance: '+str(path))
    declared_filter=descriptor.get('fixed_wrist_contact_filter')
    profile_filter=spec.get('carry_profile',{}).get('fixed_wrist_contact_filter')
    if (declared_filter!=profile_filter
            or not posture and ('fixed_wrist_contact_filter' in descriptor or profile_filter is not None)):
        raise ValueError('Candidate filter requires explicit V2 descriptor matching the audited profile')
    _validate_control(directory, spec, frozen,profile,posture=posture,
                      serial_effort_profile=serial_effort_profile)
    result=dict(selection='evaluated_posture_continuation' if posture else 'evaluated_continuation', package_directory=str(directory),
        artifact_sha256=hashes, profile_directory=str(profile), profile_sha256=profile_hashes,
        historically_trained_on=history, descriptor_path=str(descriptor_path),
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        provenance=provenance, evaluation_evidence=evaluations)
    if posture:result['profile_handoff_max_error_rad']=.1
    if declared_filter is not None:result['fixed_wrist_contact_filter']=declared_filter
    if serial_effort_profile is not None:result[SERIAL_EFFORT_FIELD]=serial_effort_profile
    return result


def require_posture_domain_diagnostic(enabled,selection,*,transport,current_grasp_reference):
    """An execution option, never a change to the pinned policy/profile contract."""
    if type(enabled) is not bool:
        raise ValueError('Posture domain diagnostic must be bool')
    if enabled and (not isinstance(selection,dict)
            or selection.get('selection')!='evaluated_posture_continuation'
            or selection.get('profile_handoff_max_error_rad')!=.1
            or transport is not True or current_grasp_reference is not True):
        raise ValueError('Posture domain diagnostic requires validated V2 transport with current grasp')


def check_posture_handoff(selection,hold_q,accepted_target,*,posture_domain_diagnostic=False):
    """V2-only pre-LOW domain gate; legacy current-grasp diagnostics are unchanged."""
    if type(posture_domain_diagnostic) is not bool:
        raise ValueError('Posture domain diagnostic must be bool')
    if posture_domain_diagnostic and selection.get('selection')!='evaluated_posture_continuation':
        raise ValueError('Posture domain diagnostic requires validated V2 selection')
    if selection.get('selection')!='evaluated_posture_continuation':return None
    if selection.get('profile_handoff_max_error_rad')!=.1:
        raise ValueError('Invalid posture handoff contract')
    arrays=[np.asarray(v,dtype=float)for v in (hold_q,accepted_target)]
    if any(x.shape!=(31,)or not np.isfinite(x).all()for x in arrays):
        raise ValueError('Finite31-axis posture handoff references required')
    indices=[15,16,17,18,22,23,24,25]
    with np.errstate(over='ignore',invalid='ignore'):
        delta=arrays[1][indices]-arrays[0][indices]
    if not np.isfinite(delta).all():
        raise ValueError('Finite held8 posture difference required')
    record=dict(sdk_indices=indices,profile_target_rad=arrays[0][indices].tolist(),
        accepted_target_rad=arrays[1][indices].tolist(),delta_rad=delta.tolist(),
        max_abs_error_rad=float(np.max(abs(delta))),limit_rad=.1)
    record.update(profile_match=record['max_abs_error_rad']<=.1,
                  posture_domain_diagnostic=posture_domain_diagnostic,runtime_domain_validated=False)
    if not record['profile_match'] and not posture_domain_diagnostic:
        raise ValueError('Acquired posture differs from evaluated profile by more than0.1rad: '+json.dumps(record))
    return record


def validate_runtime_selection(scope):
    """Independently re-read pins; never trust a host-supplied package or hash alone."""
    recorded = scope.get('policy_selection')
    if recorded is None:
        actual=resolve_package()  # Historical frozen manifests remain valid.
        require_posture_domain_diagnostic(scope.get('posture_domain_diagnostic',False),actual,
            transport=scope.get('scope')=='loaded_transport',current_grasp_reference=scope.get('current_grasp_reference'))
        return actual
    if not isinstance(recorded, dict): raise ValueError('Invalid loaded policy selection')
    kind = recorded.get('selection')
    if kind == 'frozen1750_profile76':
        actual = resolve_package()
    elif kind in ('evaluated_continuation','evaluated_posture_continuation'):
        transport=scope.get('scope')=='loaded_transport'
        if transport and (scope.get('duration_s')!=180.
                or not isinstance(scope.get('transport'),dict)
                or scope['transport'].get('schema')!='htd_loaded_transport/v1'
                or not isinstance(scope.get('grasp_retention'),dict)):
            raise ValueError('Candidate transport requires explicit bounded route and grasp-retention scope')
        require_candidate_scope(recorded.get('package_directory'), enabled=transport or scope.get('scope') == 'loaded_goal_hold_only',
            current_grasp_reference=scope.get('current_grasp_reference'),
            command_observer=isinstance(scope.get('command_observer'), dict),
            wrist_mode_entry=isinstance(scope.get('wrist_mode_entry'), dict),
            goal_hold=isinstance(scope.get('goal_hold'), dict))
        if not recorded.get('package_directory'): raise ValueError('Missing candidate selection directory')
        actual = resolve_package(recorded['package_directory'])
    else:
        raise ValueError('Unknown loaded policy selection')
    if recorded != actual or scope.get('historically_trained_on') != actual['historically_trained_on']:
        raise ValueError('Loaded policy selection/descriptor changed since setup')
    require_posture_domain_diagnostic(scope.get('posture_domain_diagnostic',False),actual,
        transport=scope.get('scope')=='loaded_transport',current_grasp_reference=scope.get('current_grasp_reference'))
    return actual
