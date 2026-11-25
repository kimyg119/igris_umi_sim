"""CPU checks of the selected v7 actor against an independent Lab3 assembly.

These tests consume the supplied selected policy and checkpoint; they do not
start Isaac/MuJoCo or publish SDK commands.
"""
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SELECTED = Path(__file__).resolve().parents[1] / "igris-c-wbc-base/evidence/htd_xy_20260918/selected"
POLICY_SLOTS = list(range(3, 15)) + [0, 1, 2]
DEFAULT_Q = np.array([
    0, 0, 0, -.2, 0, 0, .3, -.1, 0, -.2, 0, 0, .3, -.1, 0,
    .13, .13, 0, -.3, 0, 0, 0, .13, -.13, 0, -.3, 0, 0, 0, 0, 0,
], dtype=np.float32)


def carry_v2_spec():
    spec=json.loads((SELECTED/'deployment.json').read_text())
    names=spec['all_joint_names'];defaults=dict(zip(names,spec['default_joint_pos']))
    defaults['Joint_Waist_Yaw']=float(np.float32(.75566676))
    defaults['Joint_Waist_Pitch']=float(np.float32(.01830422))
    spec['default_joint_pos']=[defaults[n] for n in names]
    hashes={'report.json':'a'*64,'carry_joint_targets.json':'b'*64,'carry_load_proxy.usda':'c'*64}
    spec['carry_profile']={'schema':'htd_carry_runtime/v2','profile_sha256':hashes.copy(),
        'policy_joint_offsets':{n:defaults[n] for n in spec['policy_joint_names']},
        'held_joint_targets_rad':{n:defaults[n] for n in names if n not in spec['policy_joint_names']},
        'numerical_contract':{'schema':'htd_carry_numerical_contract/v2',
            'relative_joint_position_origin':'policy_joint_offsets_rad',
            'action_target_rule':'policy_joint_offsets_rad + 0.25 * action'}}
    spec['contract']['action_offset_source']='explicit_native_carry_profile/v2'
    spec['contract']['action_offset_profile_sha256']=hashes.copy()
    return spec


def test_explicit_carry_v2_contract_accepts_matching_offsets():
    importlib.import_module('htd_policy').HTDPolicy._validate_contract(carry_v2_spec())


@pytest.mark.parametrize('tamper',['manufacturer_label','offset','profile_hash','origin'])
def test_explicit_carry_v2_contract_rejects_ambiguous_offsets(tamper):
    spec=carry_v2_spec()
    if tamper=='manufacturer_label':spec['contract']['action_offset_source']='manufacturer_public_default_pose'
    elif tamper=='offset':spec['carry_profile']['policy_joint_offsets']['Joint_Waist_Yaw']+=.01
    elif tamper=='profile_hash':spec['contract']['action_offset_profile_sha256']['report.json']='d'*64
    else:spec['carry_profile']['numerical_contract']['relative_joint_position_origin']='unrecorded_bias'
    with pytest.raises(ValueError):importlib.import_module('htd_policy').HTDPolicy._validate_contract(spec)


def adapter(**kwargs):
    assert importlib.util.find_spec("htd_policy") is not None, "HTDPolicy adapter has not been implemented"
    return importlib.import_module("htd_policy").HTDPolicy(**kwargs)


def sample(sim_time=0., **overrides):
    values = dict(q31=DEFAULT_Q.copy(), dq31=np.zeros(31),
                  torso_quat_wxyz=np.array([1., 0., 0., 0.]),
                  torso_gyro=np.zeros(3), command_xyz=np.zeros(3), sim_time=sim_time)
    values.update(overrides)
    return values


class Lab3Reference:
    """Independent torch implementation of no-noise Lab3 observation/action.

    Uses trainer float32 clock semantics, and reconstructs the original actor
    from model.pt rather than invoking the adapter's TorchScript/helper code.
    """
    def __init__(self):
        self.weights = torch.load(SELECTED / "model.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
        self.actions = torch.zeros(1, 15)
        self.clock = torch.zeros(1)
        self.histories = None

    def step(self, values):
        q = torch.tensor(values["q31"], dtype=torch.float32)[POLICY_SLOTS][None]
        dq = torch.tensor(values["dq31"], dtype=torch.float32)[POLICY_SLOTS][None]
        command = torch.tensor(values["command_xyz"], dtype=torch.float32)[None]
        moving = ((torch.linalg.vector_norm(command[:, :2], dim=-1) > .005)
                  | (command[:, 2].abs() > .01))
        # Rotation-matrix projection is independent of adapter's scalar formula.
        quat = np.array(values["torso_quat_wxyz"], dtype=np.float64)
        w, x, y, z = quat / np.linalg.norm(quat)
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                             [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                             [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        gravity = torch.tensor(rotation.T @ np.array([0., 0., -1.]), dtype=torch.float32)[None]
        phase = (self.clock / .7) % 1.
        frames = [torch.tensor(values["torso_gyro"], dtype=torch.float32)[None], gravity,
                  torch.cat((command, torch.tensor([[.94, 0., 0., 0.]])), -1),
                  q-torch.from_numpy(DEFAULT_Q[POLICY_SLOTS])[None], dq, self.actions,
                  torch.stack(((2*torch.pi*phase).sin(), (2*torch.pi*phase).cos()), -1)*moving[:, None]]
        if self.histories is None:
            self.histories = [torch.stack((frame, frame), dim=1) for frame in frames]
        else:
            self.histories = [torch.stack((history[:, 1], frame), dim=1)
                              for history, frame in zip(self.histories, frames)]
        observation = torch.cat([history.flatten(1) for history in self.histories], -1)
        weights = self.weights
        hidden = (observation-weights["obs_normalizer._mean"]) / (weights["obs_normalizer._std"]+.01)
        for layer in (0, 2, 4, 6):
            hidden = torch.nn.functional.linear(hidden, weights[f"mlp.{layer}.weight"], weights[f"mlp.{layer}.bias"])
            if layer != 6:
                hidden = torch.nn.functional.elu(hidden)
        self.actions = hidden.detach()
        target = DEFAULT_Q.copy()
        target[POLICY_SLOTS] += .25*self.actions[0].numpy()
        self.clock = torch.where(moving, self.clock+.02, 0.)
        return observation[0].numpy(), self.actions[0].numpy(), target


def test_actual_policy_matches_lab3_and_original_checkpoint_across_phase_transitions():
    policy, reference = adapter(), Lab3Reference()
    rng = np.random.default_rng(20920)
    max_action = 0.
    for index in range(90):
        # Includes yaw-only, exact standing thresholds, diagonal movement and restart.
        command = ([.06, -.02, .1] if index < 38 else [0, 0, -.02] if index < 43
                   else [.005, 0, .01] if index < 48 else [.003, .004, 0] if index < 53
                   else [0, 0, 0] if index < 58 else [.04, .02, 0])
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        values = sample(4.+index*.02 + (index % 4)*.0004,
                        q31=DEFAULT_Q+rng.normal(0, .08, 31), dq31=rng.normal(0, .25, 31),
                        torso_quat_wxyz=quat, torso_gyro=rng.normal(0, .2, 3), command_xyz=command)
        result = policy.step(**values)
        observation, action, target = reference.step(values)
        np.testing.assert_allclose(result.observation, observation, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(result.raw_action, action, atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(result.q_target, target, atol=1e-6, rtol=2e-6)
        assert result.scheduled_policy_time == pytest.approx(4.+index*.02)
        assert result.observation_sim_time == values["sim_time"]
        assert result.lag_s == pytest.approx((index % 4)*.0004)
        assert result.policy_time == result.scheduled_policy_time
        max_action = max(max_action, float(np.abs(action).max()))
    assert max_action > 1., "Fixture must exercise unclipped previous raw actions"


def test_sdk_mapping_default_upper_hold_and_immutable_outputs():
    policy = adapter()
    np.testing.assert_array_equal(policy.default_q, DEFAULT_Q)
    assert policy.dt == .02
    with pytest.raises(ValueError):
        policy.default_q[15] = 8
    with pytest.raises(ValueError):
        policy.default_q.setflags(write=True)
    with pytest.raises(AttributeError):
        policy.dt = .1
    result = policy.step(**sample(q31=DEFAULT_Q+1.))
    np.testing.assert_array_equal(result.q_target[15:], DEFAULT_Q[15:])
    np.testing.assert_allclose(result.q_target[POLICY_SLOTS], DEFAULT_Q[POLICY_SLOTS]+.25*result.raw_action)
    for array in (result.q_target, result.raw_action, result.observation):
        with pytest.raises(ValueError):
            array[0] = 7
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_async_deadlines_duplicate_calls_and_no_history_advance():
    policy, reference = adapter(), Lab3Reference()
    assert policy.policy_due(8.)
    first = policy.step(**sample(8., command_xyz=[.05, 0, 0]))
    reference.step(sample(8., command_xyz=[.05, 0, 0]))
    assert not policy.policy_due(8.)
    with pytest.raises(ValueError, match="due"):
        policy.step(**sample(8.))
    assert not policy.policy_due(8.019)
    with pytest.raises(ValueError, match="due"):
        policy.step(**sample(8.019))
    assert policy.policy_due(8.027)
    second = policy.step(**sample(8.027, command_xyz=[.05, 0, 0]))
    expected, action, _ = reference.step(sample(8.027, command_xyz=[.05, 0, 0]))
    np.testing.assert_allclose(second.observation, expected, atol=1e-6)
    np.testing.assert_allclose(second.raw_action, action, atol=2e-6)
    np.testing.assert_array_equal(second.observation[101:116], first.raw_action)
    assert second.scheduled_policy_time == pytest.approx(8.02)
    assert second.lag_s == pytest.approx(.007)
    assert policy.policy_due(8.04)
    third = policy.step(**sample(8.04))
    assert third.scheduled_policy_time == pytest.approx(8.04)
    assert third.lag_s == pytest.approx(0.)


def test_reset_repeats_first_frame_clears_actions_phase_and_time():
    policy = adapter()
    initial = policy.step(**sample(3., command_xyz=[.05, 0, 0]))
    policy.step(**sample(3.02, command_xyz=[.05, 0, 0]))
    policy.reset(sim_time=2.)
    assert not policy.policy_due(1.999)
    again = policy.step(**sample(2.001, command_xyz=[.05, 0, 0]))
    np.testing.assert_array_equal(again.observation, initial.observation)
    np.testing.assert_array_equal(again.raw_action, initial.raw_action)
    assert again.scheduled_policy_time == 2.
    assert again.lag_s == pytest.approx(.001)
    policy.reset()
    assert policy.step(**sample(.3)).scheduled_policy_time == .3


def test_due_rejects_backwards_clock_and_missed_full_period():
    policy = adapter()
    policy.step(**sample(10.))
    assert not policy.policy_due(10.01)
    with pytest.raises(ValueError, match="backward"):
        policy.policy_due(10.009)
    with pytest.raises(ValueError, match="backward"):
        policy.step(**sample(9.))
    with pytest.raises(ValueError, match="missed"):
        policy.policy_due(10.04)
    with pytest.raises(ValueError, match="missed"):
        policy.step(**sample(10.05))
    policy.reset(sim_time=4.)
    with pytest.raises(ValueError, match="missed"):
        policy.step(**sample(4.02))


def test_phase_uses_float32_training_clock_over_many_periods():
    policy, reference = adapter(), Lab3Reference()
    for index in range(400):
        values = sample(index*.02, command_xyz=[.03, -.02, .01])
        result = policy.step(**values)
        observation, _, _ = reference.step(values)
        np.testing.assert_allclose(result.observation[-4:], observation[-4:], atol=5e-7)


@pytest.mark.parametrize("field,value", [
    ("q31", np.zeros(30)), ("q31", np.full(31, np.nan)),
    ("dq31", np.zeros((1, 31))), ("dq31", np.full(31, np.inf)),
    ("torso_gyro", [0, 0]), ("torso_gyro", [0, np.nan, 0]),
    ("torso_quat_wxyz", [0, 0, 0, 0]), ("torso_quat_wxyz", [2, 0, 0, 0]),
    ("torso_quat_wxyz", [np.nan, 0, 0, 1]), ("torso_quat_wxyz", [1, 0, 0]),
    ("command_xyz", [0, 0]), ("command_xyz", [0, np.inf, 0]),
    ("sim_time", np.nan), ("sim_time", np.inf), ("sim_time", -1.),
])
def test_invalid_input_does_not_consume_first_observation(field, value):
    policy = adapter()
    with pytest.raises(ValueError):
        policy.step(**sample(**{field: value}))
    result = policy.step(**sample())
    observation, action, _ = Lab3Reference().step(sample())
    np.testing.assert_allclose(result.observation, observation, atol=1e-6)
    np.testing.assert_allclose(result.raw_action, action, atol=2e-6)


def test_hash_and_incompatible_contract_fail_before_inference(tmp_path):
    corrupt = tmp_path / "policy.ts"
    corrupt.write_bytes((SELECTED / "policy.ts").read_bytes()+b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter(policy_path=corrupt)
    contract = json.loads((SELECTED / "deployment.json").read_text())
    contract["input_shape"] = [None, 116]
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="contract"):
        adapter(deployment_path=deployment)
