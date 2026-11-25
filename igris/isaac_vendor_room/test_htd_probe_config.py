"""Explicit artifact/hold routing is offline; these tests never start native ROS."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/hlik_transport_20260918_2005'
V2=P/'htd_carry_v2_selected01'
HOLD=P/'htd_carry_profile_v2_01/carry_joint_targets.json'


def config(**kwargs):
    return importlib.import_module('htd_probe_config').resolve_artifacts(**kwargs)


def v2(**kwargs):
    options=dict(policy_path=V2/'policy.ts',deployment_path=V2/'deployment.json',
                 training_source=V2/'training_env.yaml',native_hold_source=HOLD)
    options.update(kwargs)
    return config(**options)


def test_original_defaults_and_raw_target_are_preserved():
    from htd_policy import HTDPolicy,SELECTED
    a=config();p=HTDPolicy()
    assert a.policy_path==SELECTED/'policy.ts' and not a.requires_initial_state
    np.testing.assert_array_equal(a.hold_q,p.default_q)
    raw=p.default_q.copy();raw[:15]+=.01
    np.testing.assert_array_equal(a.merge_actor_target(raw),raw)


@pytest.mark.parametrize('field',['policy_path','deployment_path','training_source'])
def test_partial_artifact_override_fails_closed(field):
    with pytest.raises(ValueError,match='together'):
        config(**{field:V2/'policy.ts'})


def test_v2_preserves_policy15_and_captured_wrist_commands_not_measured_outputs():
    from htd_policy import HTDPolicy,POLICY_SDK_INDICES
    a=v2();p=HTDPolicy(a.policy_path,a.deployment_path);hold=json.loads(HOLD.read_text())
    assert a.requires_initial_state
    np.testing.assert_array_equal(a.hold_q[list(POLICY_SDK_INDICES)],p.default_q[list(POLICY_SDK_INDICES)])
    np.testing.assert_allclose(a.hold_q[19:22],hold['native_wrist_command_rad']['left'],rtol=0,atol=1e-8)
    assert not np.allclose(a.hold_q[19:22],hold['native_wrist_physical_rad']['left'])
    raw=p.default_q.copy();raw[list(POLICY_SDK_INDICES)]+=.123;before=raw.copy()
    merged=a.merge_actor_target(raw)
    np.testing.assert_array_equal(merged[list(POLICY_SDK_INDICES)],raw[list(POLICY_SDK_INDICES)])
    np.testing.assert_array_equal(merged[15:],a.hold_q[15:]);np.testing.assert_array_equal(raw,before)
    with pytest.raises(ValueError):a.hold_q[19]=0


def test_v2_requires_hashed_hold_source_and_matching_training_defaults(tmp_path):
    with pytest.raises(ValueError,match='hold'):
        v2(native_hold_source=None)
    copied=tmp_path/'hold.json';copied.write_text(HOLD.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):
        v2(native_hold_source=copied)
    from htd_policy import SELECTED
    with pytest.raises(ValueError,match='training.*default'):
        v2(training_source=SELECTED/'training_env.yaml')


def test_explicit_gain_and_effort_source_reaches_native_and_serial_pd(tmp_path):
    from htd_native_probe import load_gains,TargetLimits
    from htd_serial_pd import profile,SerialPD
    source=V2/'training_env.yaml';data=yaml.load(source.read_text(),Loader=yaml.BaseLoader)
    group=data['robot_cfg']['actuators']['X12-150']
    group['stiffness']={k:str(float(v)*1.1) for k,v in group['stiffness'].items()}
    changed=tmp_path/'training.yaml';changed.write_text(yaml.safe_dump(data))
    old=profile(source);new=profile(changed)
    assert new[0][3]==pytest.approx(old[0][3]*1.1)
    kp,kd,e=load_gains(training_source=changed)
    assert kp[3]==new[0][3] and e['training_source']==str(changed)
    assert e['training_sha256']==new[-1]['training_sha256']
    a=v2();pd=SerialPD(a.hold_q,kp,kd,TargetLimits(),delay_steps=[0,0,0,0],training_source=changed)
    assert pd.contract['training_sha256']==new[-1]['training_sha256']
    np.testing.assert_array_equal(pd.contract['feasible_command_q'],a.hold_q)
    pd.reset(1.)
    state=dict(q_output=a.hold_q.copy(),dq_output=np.zeros(31),q_encoder=a.hold_q.copy(),
               dq_encoder=np.zeros(31),sim=1.,monotonic=10.,sequence=2)
    raw=a.hold_q.copy();raw[3]+=.02
    result=pd.step(raw,state,native_time=1.,wall_time=10.)
    assert result.tau[3]==pytest.approx(new[0][3]*.02)
    np.testing.assert_array_equal(result.q[15:],a.hold_q[15:])
    np.testing.assert_array_equal(result.dq[15:],0.)
    np.testing.assert_array_equal(result.tau[15:],0.)
    np.testing.assert_array_equal(result.kp[15:],kp[15:])
    np.testing.assert_array_equal(result.kd[15:],kd[15:])


def test_v2_requires_validated_initializer_and_manifest_matches_resolved_artifacts():
    from htd_native_probe import require_initial_state, require_artifact_manifest
    a=v2()
    with pytest.raises(ValueError,match='initial state'):
        require_initial_state(a,None)
    require_initial_state(config(),None)
    require_initial_state(a,{'schema':'native_empty_initial_state/v1'})
    require_artifact_manifest(a,{'htd_artifacts':a.evidence})
    stale=dict(a.evidence);stale['sha256']=dict(stale['sha256'],training='changed')
    with pytest.raises(ValueError,match='artifact'):
        require_artifact_manifest(a,{'htd_artifacts':stale})


def test_v2_launch_refuses_absent_initializer_before_creating_scene(tmp_path):
    from types import SimpleNamespace
    from htd_native_probe import launch
    args=SimpleNamespace(policy_path=V2/'policy.ts',deployment_path=V2/'deployment.json',
                         training_gains_source=V2/'training_env.yaml',native_hold_source=HOLD,
                         initial_state_file=None)
    with pytest.raises(ValueError,match='initial-state-file'):
        launch(tmp_path/'never_created',args)
    assert not (tmp_path/'never_created').exists()
