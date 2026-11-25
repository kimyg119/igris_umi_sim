"""CPU inference adapter for the selected, simulation-only HTD Lab3 v7 actor.

Inputs are native SDK31 serial joint coordinates, torso body angular velocity,
and torso world quaternion in wxyz order. ``command_xyz`` means pelvis-heading
velocity [vx (m/s), vy (m/s), wz (rad/s)], not a world position target. Upstream
IMU mounting conversion and world-XY feedback belong to the caller.

Call ``policy_due(sim_time)`` for each new native observation, then ``step`` only
when due; otherwise hold the last result. The first step establishes the clock
origin unless ``reset(sim_time=...)`` supplies one. Deadlines are origin+k*20ms.
A sample may be late by less than 20ms, and its lag is recorded. A full missed
period raises rather than duplicating inference, replaying observations, or
changing the phase clock. Duplicate/early steps also raise without consuming
history. Reset explicitly after a simulation restart or discontinuity.

The result is an unconstrained trained target (default + .25*raw action). Any
caller-side clipping must not replace raw-action history. Arm override, joint
limits, PD gains, native motor mapping, and SDK publishing are outside this
adapter. This module never starts a simulator or connects to robot hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SELECTED = Path(__file__).resolve().parents[1] / "igris-c-wbc-base/evidence/htd_xy_20260918/selected"
SDK_NAMES = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch") + tuple(
    f"Joint_{joint}_{side}" for side in ("Left", "Right")
    for joint in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
) + tuple(
    f"Joint_{joint}_{side}" for side in ("Left", "Right")
    for joint in ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch")
) + ("Joint_Neck_Yaw", "Joint_Neck_Pitch")
POLICY_SDK_INDICES = tuple(range(3, 15)) + (0, 1, 2)
OBSERVATION_TERMS = (
    ("base_ang_vel", 3), ("projected_gravity", 3), ("command", 7),
    ("joint_pos_rel", 15), ("joint_vel", 15), ("previous_raw_action", 15),
    ("gait_phase", 2),
)
_TIME_TOLERANCE = 1e-9


def _immutable(values: np.ndarray) -> np.ndarray:
    """Bytes-backed arrays cannot have their writeable flag re-enabled."""
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _vector(values, size: int, name: str, *, dtype=np.float32) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite {size}-value vector") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-value vector")
    return result


def _time(value) -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sim_time must be a finite nonnegative scalar") from exc
    if array.shape != () or not np.isfinite(array) or array < 0:
        raise ValueError("sim_time must be a finite nonnegative scalar")
    return float(array)


@dataclass(frozen=True)
class HTDPolicyStep:
    q_target: np.ndarray
    raw_action: np.ndarray
    observation: np.ndarray
    scheduled_policy_time: float
    observation_sim_time: float
    lag_s: float

    @property
    def policy_time(self) -> float:
        return self.scheduled_policy_time


class HTDPolicy:
    """Hash-verified actor plus v7 observation, history, and policy-clock state."""

    def __init__(self, policy_path=SELECTED / "policy.ts", deployment_path=SELECTED / "deployment.json"):
        self.policy_path = Path(policy_path)
        self.deployment_path = Path(deployment_path)
        spec = json.loads(self.deployment_path.read_text())
        self._validate_contract(spec)
        self.policy_sha256 = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        if self.policy_sha256 != spec.get("policy_sha256"):
            raise ValueError("Policy SHA-256 differs from deployment.json")
        defaults = _vector(spec["default_joint_pos"], 23, "default_joint_pos")
        q = np.zeros(31, dtype=np.float32)
        for name, value in zip(spec["all_joint_names"], defaults):
            q[SDK_NAMES.index(name)] = value
        self._default_q = _immutable(q)
        self._slots = list(POLICY_SDK_INDICES)
        self._actor = torch.jit.load(str(self.policy_path), map_location="cpu").eval()
        # Validate the artifact's callable boundary without advancing adapter state.
        self._infer(np.zeros(120, dtype=np.float32))
        self.reset()

    @staticmethod
    def _validate_contract(spec):
        expected = {
            "schema": "wbc_world_xy_lab3_deployment/v1",
            "input_shape": [None, 120], "output_shape": [None, 15],
            "phase_gait": True, "phase_period_s": .7,
            "history_layout": "term-major; oldest then newest; repeat first on reset",
            "observation_terms": [list(term) for term in OBSERVATION_TERMS],
            "policy_joint_names": [SDK_NAMES[index] for index in POLICY_SDK_INDICES],
        }
        for name, value in expected.items():
            if spec.get(name) != value:
                raise ValueError(f"Unsupported HTD v7 contract: {name}")
        contract = spec.get("contract", {})
        for name, value in {
            "articulation_root": "base_link", "root_semantic": "torso",
            "pelvis_body": "Link_Waist_Pitch", "quaternion_order": "wxyz",
            "physics_dt_s": .005, "policy_dt_s": .02, "action_scale": .25,
            "history_length": 2, "history_reset_semantic": "repeat_first_observation",
            "command_body_height_m": .94,
        }.items():
            if contract.get(name) != value:
                raise ValueError(f"Unsupported HTD v7 contract: {name}")
        names = spec.get("all_joint_names", [])
        training_names = {name for name in SDK_NAMES if "Wrist" not in name and "Neck" not in name}
        if len(names) != 23 or set(names) != training_names:
            raise ValueError("Unsupported HTD v7 contract: all_joint_names")
        profile=spec.get('carry_profile',{})
        source=contract.get('action_offset_source')
        if source=='manufacturer_public_default_pose':
            if profile.get('schema')=='htd_carry_runtime/v2':
                raise ValueError('Carry V2 offsets must be identified explicitly')
        elif source=='explicit_native_carry_profile/v2':
            numerical=profile.get('numerical_contract',{})
            expected_numerical={'schema':'htd_carry_numerical_contract/v2',
                'relative_joint_position_origin':'policy_joint_offsets_rad',
                'action_target_rule':'policy_joint_offsets_rad + 0.25 * action'}
            if (profile.get('schema')!='htd_carry_runtime/v2'
                    or any(numerical.get(k)!=v for k,v in expected_numerical.items())
                    or not profile.get('profile_sha256')
                    or contract.get('action_offset_profile_sha256')!=profile['profile_sha256']):
                raise ValueError('Carry V2 numerical contract/profile identity mismatch')
            offsets=profile.get('policy_joint_offsets',{})
            held=profile.get('held_joint_targets_rad',{})
            if set(offsets)!=set(spec['policy_joint_names']) or set(held)!=training_names-set(offsets):
                raise ValueError('Carry V2 must specify exactly the policy and held joint defaults')
            declared={**offsets,**held}
            values=_vector([declared[n] for n in names],23,'Carry V2 defaults')
            actual=_vector(spec.get('default_joint_pos'),23,'Exported defaults')
            if not np.array_equal(values,actual):
                raise ValueError('Carry V2 exported defaults differ from the declared offsets')
        else:
            raise ValueError('Unsupported HTD action offset source')

    @property
    def default_q(self) -> np.ndarray:
        """Immutable SDK31 manufacturer defaults; absent wrists and neck are zero."""
        return self._default_q

    @property
    def dt(self) -> float:
        return .02

    def reset(self, sim_time=None) -> None:
        origin = None if sim_time is None else _time(sim_time)
        self._origin = origin
        self._step_index = 0
        self._last_seen_time = None
        self._previous_frames = None
        self._previous_raw_action = np.zeros(15, dtype=np.float32)
        self._gait_clock = torch.zeros(1, dtype=torch.float32)

    def _check_due(self, sim_time) -> tuple[float, bool, float]:
        timestamp = _time(sim_time)
        if self._last_seen_time is not None and timestamp < self._last_seen_time:
            raise ValueError("Simulation time moved backward; reset the HTD policy")
        deadline = timestamp if self._origin is None else self._origin+self._step_index*self.dt
        lag = timestamp-deadline
        if lag >= self.dt-_TIME_TOLERANCE:
            raise ValueError("HTD policy missed an entire 20 ms period; reset explicitly")
        return timestamp, lag >= -_TIME_TOLERANCE, deadline

    def policy_due(self, sim_time) -> bool:
        """Check the next deadline; rejects backwards time or a full missed tick.

        This records the observed timestamp for monotonicity, but does not change
        the deadline, action history, or phase. Rechecking the same time is safe.
        """
        timestamp, due, _ = self._check_due(sim_time)
        self._last_seen_time = timestamp
        return due

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            result = self._actor(torch.from_numpy(observation[None].copy()))
        if (not isinstance(result, torch.Tensor) or result.device.type != "cpu"
                or tuple(result.shape) != (1, 15) or not torch.isfinite(result).all()):
            raise ValueError("HTD actor must return finite CPU actions with shape (1, 15)")
        return result[0].numpy().astype(np.float32, copy=True)

    def step(self, q31, dq31, torso_quat_wxyz, torso_gyro, command_xyz, sim_time) -> HTDPolicyStep:
        """Infer once at a due timestamp; validate fully before consuming state."""
        timestamp, due, deadline = self._check_due(sim_time)
        if not due:
            raise ValueError("HTD policy is not due; hold the previous target")
        q = _vector(q31, 31, "q31")
        dq = _vector(dq31, 31, "dq31")
        gyro = _vector(torso_gyro, 3, "torso_gyro")
        command = _vector(command_xyz, 3, "command_xyz")
        quat = _vector(torso_quat_wxyz, 4, "torso_quat_wxyz", dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or abs(norm-1.) > 1e-3:
            raise ValueError("torso_quat_wxyz must be a unit quaternion (tolerance 1e-3)")
        w, x, y, z = quat/norm
        gravity = np.array([2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))], dtype=np.float32)
        command_tensor = torch.from_numpy(command.copy())
        moving = bool((torch.linalg.vector_norm(command_tensor[:2]) > .005)
                      | (command_tensor[2].abs() > .01))
        phase = (self._gait_clock/.7) % 1.
        phase_features = torch.stack(((2*torch.pi*phase).sin(), (2*torch.pi*phase).cos()), -1)
        phase_features *= moving
        frames = [gyro, gravity, np.concatenate((command, np.array([.94, 0., 0., 0.], dtype=np.float32))),
                  q[self._slots]-self._default_q[self._slots], dq[self._slots],
                  self._previous_raw_action, phase_features[0].numpy()]
        previous = frames if self._previous_frames is None else self._previous_frames
        observation = np.concatenate([part for old, new in zip(previous, frames) for part in (old, new)])
        if not np.isfinite(observation).all():
            raise ValueError("HTD observation overflowed")
        action = self._infer(observation)
        target = self._default_q.copy()
        target[self._slots] += .25*action
        if not np.isfinite(target).all():
            raise ValueError("HTD target overflowed")
        result = HTDPolicyStep(_immutable(target), _immutable(action), _immutable(observation),
                               deadline, timestamp, max(0., timestamp-deadline))
        # Commit only after successful validation/inference; float32 accumulation
        # matches WorldXYLab3Env._pre_physics_step, including stand/reset behavior.
        self._previous_frames = [frame.copy() for frame in frames]
        self._previous_raw_action = action.copy()
        self._gait_clock = self._gait_clock+self.dt if moving else torch.zeros_like(self._gait_clock)
        self._origin = timestamp if self._origin is None else self._origin
        self._step_index += 1
        self._last_seen_time = timestamp
        return result
