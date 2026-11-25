"""Opt-in SIM serial PD effort adapter; no ROS, simulator or device operations.

The actor keeps its 20 ms raw reference/history. This adapter has independent
5 ms deadlines and computes from synchronized physical output q/dq. Published
q stays at the explicit feasible hold: owned axes have zero local gains,
so this q is not the PD reference. Native PJS torque/Jacobian and SafetyModule
motor limits remain downstream. Finite-rate feedback is not exact native/Isaac
actuator equivalence; callers must log and guard the actual publication time.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import yaml

from htd_policy import SDK_NAMES, SELECTED, POLICY_SDK_INDICES, _immutable

ROOT = Path(__file__).resolve().parents[2]
DELAY_GROUPS = ('X12-150', 'X8-120', 'X6-60', 'PLA')
LIBRARY_SHA256 = 'a171937ca3e6081938c606017477a72ebb7209bf8ee252bc3be7e60378e6a562'
EPS = 1e-9


def warm_policy_actor(policy):
    """Pay TorchScript graph warmup before services without touching history."""
    for _ in range(3):
        policy._infer(np.zeros(120, dtype=np.float32))


def vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (31,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite SDK31')
    return result


def profile(training_source=SELECTED/'training_env.yaml'):
    """Resolve exact selected serial gains/effort caps, never mapped motor gains."""
    source = Path(training_source)
    groups = yaml.load(source.read_text(), Loader=yaml.BaseLoader)['robot_cfg']['actuators']
    if set(groups) != set(DELAY_GROUPS):
        raise ValueError('Unknown training actuator groups')
    kp, kd, effort = np.zeros(31), np.zeros(31), np.zeros(31)
    group_indices = np.full(31, -1, dtype=int)
    def value(spec, name):
        matches = ([float(v) for pattern, v in spec.items() if re.fullmatch(pattern, name)]
                   if isinstance(spec, dict) else [float(spec)])
        if len(matches) != 1 or not np.isfinite(matches[0]) or matches[0] <= 0:
            raise ValueError('Unknown/nonpositive training value: '+name)
        return matches[0]
    for index in POLICY_SDK_INDICES:
        name = SDK_NAMES[index]
        matches = [(g, group) for g, group in groups.items()
                   if any(re.fullmatch(pattern, name) for pattern in group['joint_names_expr'])]
        if len(matches) != 1:
            raise ValueError('Ambiguous training actuator: '+name)
        key, group = matches[0]
        if (group['class_type'] != 'isaaclab.actuators.actuator_pd:DelayedPDActuator'
                or [int(group['min_delay']), int(group['max_delay'])] != [0, 2]):
            raise ValueError('Unsupported training actuator/delay contract')
        kp[index], kd[index] = value(group['stiffness'], name), value(group['damping'], name)
        effort[index] = value(group['effort_limit'], name)
        if effort[index] != value(group['effort_limit_sim'], name):
            raise ValueError('Training explicit/simulator effort caps differ')
        group_indices[index] = DELAY_GROUPS.index(key)
    params_path = ROOT/'igris/configs/controller/params.yaml'
    params = yaml.safe_load(params_path.read_text())['igris_c_param_server']['ros__parameters']
    caps = {key: vector(params[key], key) for key in
            ('torque_min', 'torque_max', 'motor_torque_min', 'motor_torque_max')}
    if any(np.any(caps[lo] >= 0) or np.any(caps[hi] <= 0)
           for lo, hi in [('torque_min', 'torque_max'), ('motor_torque_min', 'motor_torque_max')]):
        raise ValueError('Invalid native torque limits')
    library = ROOT/'igris/ws/install/igris_c_lib/lib/libigris_c_lib.so'
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise ValueError('Serial torque requires the audited vendor library')
    evidence = dict(training_source=str(source), training_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    params_source=str(params_path), params_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
                    vendor_library_sha256=LIBRARY_SHA256,
                    serial_kp=kp.tolist(), serial_kd=kd.tolist(), training_effort=effort.tolist(),
                    native_caps={key: cap.tolist() for key, cap in caps.items()})
    return kp, kd, effort, group_indices, caps, evidence


def write_motor_fields(message, record):
    """Fill all motor fields; caller checks freshness immediately before publish."""
    if len(message.motors) != 31:
        raise ValueError('LowCmd requires exactly 31 motors')
    for index, motor in enumerate(message.motors):
        motor.id = index
        for field in ('q', 'dq', 'kp', 'kd', 'tau'):
            setattr(motor, field, float(getattr(record, field)[index]))


def zero_owned_effort(message):
    """Failure/pause cleanup must never replay a held stale feedforward torque."""
    for index in POLICY_SDK_INDICES:
        motor = message.motors[index]
        motor.kp = motor.kd = motor.dq = motor.tau = 0.


@dataclass(frozen=True)
class SerialPDStep:
    scheduled_time: float
    observation_sim_time: float
    observation_monotonic: float
    sequence: int
    lag_s: float
    native_lag_s: float
    wall_age_s: float
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    tau_raw: np.ndarray
    tau_training: np.ndarray
    raw_target: np.ndarray
    delayed_target: np.ndarray
    q_output: np.ndarray
    dq_output: np.ndarray
    q_encoder: np.ndarray
    dq_encoder: np.ndarray
    training_clipped: np.ndarray
    native_clipped: np.ndarray


class SerialPD:
    dt = .005
    max_sample_native_lag = .002
    max_sample_wall_age = .1

    def __init__(self, default_q, baseline_kp, baseline_kd, gate, *, delay_steps,
                 training_source=SELECTED/'training_env.yaml'):
        delays = np.asarray(delay_steps)
        if (delays.shape != (4,) or not np.isfinite(delays).all()
                or np.any(delays != np.floor(delays)) or np.any(delays < 0) or np.any(delays > 2)):
            raise ValueError('Explicit four actuator-group delays must be integers in [0, 2]')
        self._default = _immutable(vector(default_q, 'default_q'))
        gate.check(self._default)
        self._gate = gate
        self._owned = np.zeros(31, dtype=bool)
        self._owned[list(POLICY_SDK_INDICES)] = True
        self._kp, self._kd = vector(baseline_kp, 'baseline_kp').copy(), vector(baseline_kd, 'baseline_kd').copy()
        if np.any(self._kp < 0) or np.any(self._kd < 0):
            raise ValueError('Negative baseline hold gains')
        self._kp[self._owned] = self._kd[self._owned] = 0.
        self._train_kp, self._train_kd, self._effort, groups, self._caps, evidence = profile(training_source)
        self._delay = np.zeros(31, dtype=int)
        self._delay[self._owned] = delays.astype(int)[groups[self._owned]]
        self.contract = dict(evidence, actor_dt_s=.02, pd_dt_s=self.dt,
            delay_groups=list(DELAY_GROUPS), configured_delay_steps=delays.astype(int).tolist(),
            training_delay_range_steps=[0, 2], nominal_disables_delay=False,
            delay_semantic='Fixed explicit per-group delay in PD ticks; repeat first reference on reset',
            exact_native_actuator_equivalence=False, owned_sdk_indices=np.flatnonzero(self._owned).tolist(),
            command_kp=self._kp.tolist(), command_kd=self._kd.tolist(), feasible_command_q=self._default.tolist(),
            q_reference='Explicit feasible baseline; owned kp=kd=0, raw actor reference used only by external PD',
            max_output_native_lag_s=self.max_sample_native_lag, max_output_wall_age_s=self.max_sample_wall_age,
            output_observer_hz=300, observer_contract='Use first 300Hz synchronized sample at/after each 200Hz deadline; alternate ideal deadline lag 0/1.667ms',
            missed_period_guard_s=self.dt, position_bias_guard_rad=.10,
            stale_eligible_sample='Wait without computation/publication while wall-fresh and before the same full5ms deadline',
            downstream='Audited vendor PJS damped Jacobian torque mapping and serial/motor torque caps retained',
            limitations=['Explicit delays need not match original random group delays for a seed.',
                         'The 300Hz FK/output observer samples 2100Hz physics; 200Hz PD publications have alternating ideal intervals 6.667/3.333ms, not exact5ms.',
                         'Publication and DDS/native consumption are not synchronous.',
                         'No claim of native applied effort parity until closed-loop delivery is measured.'])
        self.reset(0.)

    def reset(self, sim_time):
        if not np.isfinite(sim_time) or sim_time < 0:
            raise ValueError('Finite nonnegative PD origin required')
        self._origin = float(sim_time)
        self.step_count = 0
        self.stale_poll_waits = 0
        self._last_sim = self._last_native = None
        self._last_sequence = 0
        self._history = []

    @property
    def next_due(self):
        return self._origin+self.step_count*self.dt

    def due(self, sample, *, native_time, wall_time):
        sim, wall = float(sample['sim']), float(sample['monotonic'])
        seq = sample['sequence']
        if not np.isfinite([sim, wall, native_time, wall_time]).all() or sim < 0:
            raise ValueError('Nonfinite/negative PD timestamp')
        if (self._last_sim is not None and sim < self._last_sim-EPS
                or self._last_native is not None and native_time < self._last_native-EPS):
            raise RuntimeError('PD time moved backwards; explicit reset required')
        if native_time-self.next_due >= self.dt-EPS:
            raise RuntimeError(f'Missed PD deadline: native={native_time:.9f}, due={self.next_due:.9f}, period=0.005')
        if native_time-sim < -EPS:
            raise RuntimeError('Future native output sample for PD')
        if not 0 <= wall_time-wall < self.max_sample_wall_age:
            raise RuntimeError('Stale/future wall output sample for PD')
        if not isinstance(seq, (int, np.integer)) or seq <= 0 or seq % 2:
            raise RuntimeError('Incomplete output sequence for PD')
        # Track observed time without consuming a deadline/delay-buffer entry.
        # A reset between two early polls must not escape monotonicity checks.
        self._last_sim, self._last_native = sim, native_time
        if sim < self.next_due-EPS:
            # motor2Joint/output observer is 300Hz, not physics'2100Hz. Waiting
            # polls may see an older sample, but cannot compute/publish from it.
            # The full missed-period and wall-freshness guards remain active.
            return False
        if native_time-sim >= self.max_sample_native_lag:
            # A new300Hz sample may still arrive before this5ms deadline ends.
            # Never compute from the stale sample or advance reference history.
            self.stale_poll_waits += 1
            return False
        if seq <= self._last_sequence:
            raise RuntimeError('Duplicate/backwards output sequence at PD deadline')
        return True

    def step(self, raw_target, sample, *, native_time, wall_time):
        if not self.due(sample, native_time=native_time, wall_time=wall_time):
            raise RuntimeError('PD not due; do not advance reference delay')
        raw = vector(raw_target, 'raw_target').copy()
        q, dq, encoder, dencoder = [vector(sample[key], key) for key in
                                   ('q_output', 'dq_output', 'q_encoder', 'dq_encoder')]
        if not np.array_equal(raw[~self._owned], self._default[~self._owned]):
            raise ValueError('Nonowned actor targets must stay at exact baseline')
        if np.max(np.abs(q[self._owned]-encoder[self._owned])) > .10+1e-12:
            raise ValueError('Owned output position bias exceeds 0.10 rad')
        actual = self._default.copy()
        actual[self._owned] = q[self._owned]
        self._gate.check(actual)
        history = ([raw.copy(), raw.copy()] if not self._history else self._history)+[raw]
        delayed = raw.copy()
        for index in np.flatnonzero(self._owned):
            delayed[index] = history[-1-self._delay[index]][index]
        tau_raw = np.zeros(31)
        tau_raw[self._owned] = (self._train_kp[self._owned]*(delayed[self._owned]-q[self._owned])
                                -self._train_kd[self._owned]*dq[self._owned])
        if not np.isfinite(tau_raw).all():
            raise ValueError('Serial PD arithmetic overflow')
        train = np.clip(tau_raw, -self._effort, self._effort)
        tau = np.clip(train, self._caps['torque_min'], self._caps['torque_max'])
        result = SerialPDStep(self.next_due, float(sample['sim']), float(sample['monotonic']),
            int(sample['sequence']), max(0., sample['sim']-self.next_due), native_time-sample['sim'],
            wall_time-sample['monotonic'], *[_immutable(x) for x in
            (self._default, np.zeros(31), self._kp, self._kd, tau, tau_raw, train, raw, delayed,
             q, dq, encoder, dencoder, train != tau_raw, tau != train)])
        self.step_count += 1
        self._last_sim, self._last_native = sample['sim'], native_time
        self._last_sequence = sample['sequence']
        self._history = [h.copy() for h in history[-2:]]
        return result

    def check_publication(self, record, *, native_time, wall_time):
        """Call immediately before publishing; late candidates must never leak."""
        if (not np.isfinite([native_time, wall_time]).all()
                or not -EPS <= native_time-record.observation_sim_time < self.max_sample_native_lag
                or not -EPS <= native_time-record.scheduled_time < self.dt-EPS
                or not 0 <= wall_time-record.observation_monotonic < self.max_sample_wall_age):
            raise RuntimeError(f'Stale/missed PD publication deadline: sample_age={native_time-record.observation_sim_time:.9f}s '
                               f'(limit<0.002), deadline_lag={native_time-record.scheduled_time:.9f}s (limit<0.005), '
                               f'wall_age={wall_time-record.observation_monotonic:.9f}s (limit<0.1)')
