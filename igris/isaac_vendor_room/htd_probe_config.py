"""Offline artifact and native hold routing; no initialization or robot commands."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from htd_policy import HTDPolicy, SELECTED, SDK_NAMES, POLICY_SDK_INDICES, _immutable


@dataclass(frozen=True)
class ProbeArtifacts:
    policy_path: Path
    deployment_path: Path
    training_source: Path
    native_hold_source: Path | None
    hold_q: np.ndarray
    requires_initial_state: bool
    evidence: dict

    def merge_actor_target(self, raw_target):
        """Preserve owned15 raw reference, including excursions; retain native16."""
        raw = np.asarray(raw_target, dtype=np.float64)
        if raw.shape != (31,) or not np.isfinite(raw).all():
            raise ValueError('Actor target must be finite SDK31')
        target = self.hold_q.copy()
        target[list(POLICY_SDK_INDICES)] = raw[list(POLICY_SDK_INDICES)]
        return target


def resolve_artifacts(*, policy_path=None, deployment_path=None, training_source=None,
                      native_hold_source=None):
    """Explicit triples prevent accidentally mixing actors, defaults and gains."""
    provided = [p is not None for p in (policy_path, deployment_path, training_source)]
    if any(provided) and not all(provided):
        raise ValueError('Policy, deployment and training source must be supplied together')
    policy_path, deployment_path, training_source = [Path(p).resolve() for p in (
        policy_path or SELECTED/'policy.ts', deployment_path or SELECTED/'deployment.json',
        training_source or SELECTED/'training_env.yaml')]
    spec = json.loads(deployment_path.read_text())
    HTDPolicy._validate_contract(spec)
    paths = dict(policy=policy_path, deployment=deployment_path, training=training_source)
    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != spec['policy_sha256']:
        raise ValueError('Policy hash differs from explicit deployment')
    q = np.zeros(31, dtype=np.float64)
    for name, value in zip(spec['all_joint_names'], spec['default_joint_pos']):
        q[SDK_NAMES.index(name)] = np.float32(value)
    training = yaml.load(training_source.read_text(), Loader=yaml.BaseLoader)
    positions = training['robot_cfg']['init_state']['joint_pos']
    for name, expected in zip(spec['all_joint_names'], spec['default_joint_pos']):
        matches = [float(value) for pattern, value in positions.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1 or not np.isfinite(matches[0]) or abs(matches[0]-expected) > 1e-7:
            raise ValueError('Explicit training joint default differs from deployment: '+name)
    profile = spec.get('carry_profile', {})
    is_v2 = profile.get('schema') == 'htd_carry_runtime/v2'
    hold_path = None
    if is_v2:
        if native_hold_source is None:
            raise ValueError('V2 requires its explicit native hold source')
        hold_path = Path(native_hold_source).resolve()
        if hashlib.sha256(hold_path.read_bytes()).hexdigest() != profile['profile_sha256'].get('carry_joint_targets.json'):
            raise ValueError('Native hold source hash differs from V2 deployment')
        hold = json.loads(hold_path.read_text())
        if (hold.get('schema') != 'htd_carry_hold_targets/v2' or hold.get('units') != 'rad'
                or hold.get('joint_targets_rad') != profile['held_joint_targets_rad']
                or hold.get('policy_joint_offsets_rad') != profile['policy_joint_offsets']):
            raise ValueError('Native hold source differs from V2 numerical contract')
        for side, indices in [('left', slice(19, 22)), ('right', slice(26, 29))]:
            wrist = np.asarray(hold['native_wrist_command_rad'][side], dtype=np.float64)
            if wrist.shape != (3,) or not np.isfinite(wrist).all():
                raise ValueError('Native wrist command must be finite yaw/roll/pitch')
            q[indices] = wrist
        paths['native_hold'] = hold_path
    elif native_hold_source is not None:
        raise ValueError('Explicit native hold source currently requires V2')
    evidence = dict(paths={k: str(v) for k, v in paths.items()},
                    sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
                    action_offset_source=spec['contract']['action_offset_source'],
                    requires_initial_state=is_v2, hold_q_sdk31=q.tolist(),
                    native_wrist_semantic='Captured commands, not measured physical outputs' if is_v2 else 'Original zero wrist hold',
                    neck_semantic='Original zero neck hold',
                    hands='Default MCP30 empty hands; captured wrist commands do not imply grasp')
    return ProbeArtifacts(policy_path, deployment_path, training_source, hold_path,
                          _immutable(q), is_v2, evidence)
