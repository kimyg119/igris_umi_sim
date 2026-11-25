"""Isolated native MuJoCo HTD stand/velocity probe with empty, unloaded hands.

The host creates its own private/no-network container and always stops it. The
worker requires SIM RobotState and startup gravity compensation disabled. It
defaults to SDK joint_state q/dq; optional output coordinates use synchronized
native hinges and vendor FK with bounded position/velocity correction. No real-robot
mode, arm transport, room-layout activation, or grasp claim is supported.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import tomllib
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from htd_policy import HTDPolicy, SDK_NAMES, SELECTED, POLICY_SDK_INDICES
from htd_probe_config import resolve_artifacts
from run import ROOT, HERE, JOINTS

sys.path.insert(0, str(ROOT/'igris/igris-c-wbc-base/scripts'))
from slow_policy_runtime import map_vendor_ankle_gains, torso_imu

GAINS = ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml'
DEFAULT_MOTION = ROOT/'outputs/hlik_transport_20260918_2005/ready_plan43/motion_100hz.npz'
WAIST_GAIN_LIBRARY_SHA256 = 'a171937ca3e6081938c606017477a72ebb7209bf8ee252bc3be7e60378e6a562'


def probe_artifacts(args):
    return resolve_artifacts(policy_path=args.policy_path, deployment_path=args.deployment_path,
                             training_source=args.training_gains_source,
                             native_hold_source=args.native_hold_source)


def require_initial_state(artifacts, configuration):
    if artifacts.requires_initial_state and configuration is None:
        raise ValueError('V2 native activation requires an audited explicit initial state')


def require_artifact_manifest(artifacts, manifest):
    if manifest.get('htd_artifacts') != artifacts.evidence:
        raise ValueError('Worker artifact sources differ from the launch manifest')


def torque_ready(status, *, received_at, torque_ack_at, now):
    """Fresh post-service BmsState: producer initialized AND torque-on = 3.

    Vendor commonStatusData_st::deriveBmsInitState owns this mapping. A torque
    service acknowledgement or fresh joint samples alone do not confirm it.
    """
    return (status is not None and np.isfinite([received_at, torque_ack_at, now]).all()
            and received_at > torque_ack_at and 0 <= now-received_at < .1
            and status.get('bms_init_state') == 3 and status.get('body_power') == 1
            and status.get('legs_power') == 1 and status.get('estop') == 0)


def map_vendor_waist_gains(kp, kd):
    """Pinned vendor zero-pose diagonal motor profile, not exact serial PD.

    Supplied JointManager with PJS and waist gain mapping enabled yields these
    constants for this training profile. LOW disables that internal mapping, so
    the optional probe sends pre-mapped gains, as it already does for ankles.
    Local serial stiffness residual is +6.4% roll / -6.9% pitch; finite-angle
    linkage dynamics and discarded full-matrix terms remain unmodelled.
    """
    kp, kd = np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)
    if (kp.shape != (31,) or kd.shape != (31,) or not np.isfinite(kp).all()
            or not np.isfinite(kd).all()
            or not np.allclose(kp[1:3], [660.634418, 230.579719], rtol=0, atol=1e-6)
            or not np.allclose(kd[1:3], [34.67, 12.1], rtol=0, atol=1e-6)):
        raise ValueError('Waist gains differ from the validated vendor profile')
    kp, kd = kp.copy(), kd.copy()
    kp[1:3], kd[1:3] = 107.38057936091046, 5.6351217576586912
    return kp, kd


def load_gains(*, map_waist=False, training_source=SELECTED/'training_env.yaml'):
    """Keep 23 trained gains, map validated ankles, retain saved wrist holds."""
    training_source = Path(training_source)
    training = yaml.load(training_source.read_text(), Loader=yaml.BaseLoader)
    groups = training['robot_cfg']['actuators']
    kp, kd = np.full(31, np.nan), np.full(31, np.nan)
    def value(spec, name):
        matches = [float(v) for pattern, v in spec.items() if re.fullmatch(pattern, name)] if isinstance(spec, dict) else [float(spec)]
        if len(matches) != 1:
            raise ValueError('Ambiguous training actuator gain: '+name)
        return matches[0]
    for index, name in enumerate(SDK_NAMES):
        matching = [group for group in groups.values() if any(re.fullmatch(pattern, name) for pattern in group['joint_names_expr'])]
        if matching:
            if len(matching) != 1:
                raise ValueError('Ambiguous training actuator: '+name)
            kp[index] = value(matching[0]['stiffness'], name)
            kd[index] = value(matching[0]['damping'], name)
    saved = tomllib.loads(GAINS.read_text())['motion']['gains']
    for index in (19, 20, 21, 26, 27, 28):
        name = JOINTS[index].replace('l_', 'left_', 1) if index < 22 else JOINTS[index].replace('r_', 'right_', 1)
        kp[index], kd[index] = saved[name]['kp'], saved[name]['kd']
    # Existing run.command neutral neck hold; neck is absent in trainer/TOML.
    kp[29:], kd[29:] = 30., 3.
    serial_kp, serial_kd = kp.copy(), kd.copy()
    kp, kd = map_vendor_ankle_gains(kp, kd)
    if map_waist:
        library = ROOT/'igris/ws/install/igris_c_lib/lib/libigris_c_lib.so'
        if hashlib.sha256(library.read_bytes()).hexdigest() != WAIST_GAIN_LIBRARY_SHA256:
            raise ValueError('Waist mapping requires the audited vendor library')
        kp, kd = map_vendor_waist_gains(kp, kd)
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp < 0) or np.any(kd < 0):
        raise ValueError('Incomplete finite native gain profile')
    evidence = dict(training_source=str(training_source),
                    training_sha256=hashlib.sha256(training_source.read_bytes()).hexdigest(), wrists_source=str(GAINS),
                    neck_source='igris/isaac_vendor_room/run.py: existing neutral neck kp=30 kd=3',
                    serial_kp=serial_kp.tolist(), serial_kd=serial_kd.tolist(),
                    native_kp=kp.tolist(), native_kd=kd.tolist(),
                    waist_mapping=dict(enabled=map_waist, profile='vendor zero-pose diagonal motor gains',
                        vendor_library_sha256=WAIST_GAIN_LIBRARY_SHA256,
                        source='resume_20260920/htd_contract/output_coordinate_diagnosis/vendor_gain_audit.cpp and vendor_gain_audit.log',
                        exact_serial_pd_equivalence=False,
                        local_serial_stiffness_relative_residual=[.06398, -.06860]),
                    ankle_mapping_source='igris/igris-c-wbc-base/scripts/slow_policy_runtime.py',
                    gravity_compensation=False, trained_actuator_delay_reproduced=False)
    return kp, kd, evidence


def imu_mount(path):
    root = ET.parse(path).getroot()
    body = next((body for body in root.iter('body') if body.find("./site[@name='base_link_imu_site']") is not None), None)
    if body is None or body.get('name') != 'torso':
        raise ValueError('Expected audited torso-mounted base_link_imu_site')
    site = body.find("./site[@name='base_link_imu_site']")
    quat = np.fromstring(site.get('quat', ''), sep=' ')
    if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) > 1e-3:
        raise ValueError('Invalid torso IMU mount quaternion')
    return quat/np.linalg.norm(quat)


class TargetLimits:
    def __init__(self):
        tree = ET.parse(ROOT/'igris/igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf')
        limits = [tree.find(f".//joint[@name='{name}']/limit") for name in JOINTS]
        if any(limit is None for limit in limits):
            raise ValueError('Missing SDK31 URDF limits')
        self.lower = np.array([float(limit.get('lower')) for limit in limits])
        self.upper = np.array([float(limit.get('upper')) for limit in limits])
        sys.path.insert(0, str(ROOT/'igris/igris_curobo/src'))
        from igris_ik.vendor_limits import coupled_halfspaces
        self.matrix, self.bound = coupled_halfspaces(JOINTS)

    def check(self, q):
        q = np.asarray(q)
        if q.shape != (31,) or not np.isfinite(q).all():
            raise ValueError('Expected finite SDK31 target')
        if np.any(q < self.lower-1e-6) or np.any(q > self.upper+1e-6):
            bad = np.flatnonzero((q < self.lower-1e-6) | (q > self.upper+1e-6))
            raise ValueError('Target outside URDF limits: '+','.join(JOINTS[i] for i in bad))
        if np.any(self.matrix@q > self.bound+1e-6):
            raise ValueError('Target outside vendor coupled limits')

    def project(self, q, *, max_correction_rad=.10):
        """Euclidean projection onto physical limits with an explicit budget.

        This modifies only the outgoing target. The caller keeps the actor's
        unprojected raw action in HTD history and records both target vectors.
        """
        if not np.isfinite(max_correction_rad) or not 0 < max_correction_rad <= .30:
            raise ValueError('Projection budget must be finite and within (0, 0.30] rad')
        q = np.asarray(q, dtype=float)
        if q.shape != (31,) or not np.isfinite(q).all():
            raise ValueError('Expected finite SDK31 target')
        projected = np.clip(q, self.lower, self.upper)
        if np.any(self.matrix@projected > self.bound):
            from scipy.optimize import minimize
            result = minimize(lambda x: .5*np.sum((x-q)**2), projected,
                              jac=lambda x: x-q, method='SLSQP',
                              bounds=list(zip(self.lower, self.upper)),
                              constraints={'type': 'ineq', 'fun': lambda x: self.bound-self.matrix@x,
                                           'jac': lambda x: -self.matrix},
                              options={'maxiter': 100, 'ftol': 1e-12})
            if not result.success:
                raise ValueError('Physical target projection failed: '+result.message)
            projected = result.x
        self.check(projected)
        correction = float(np.max(np.abs(projected-q)))
        if correction > max_correction_rad+1e-12:
            raise ValueError(f'Target projection exceeds {max_correction_rad:.2f} rad: {correction:.6f}')
        return projected


def require_state(environment, low_age, mode_age):
    if environment != 1:
        raise RuntimeError('SIM RobotState required; hardware is never supported')
    if not 0 <= low_age < .1 or not 0 <= mode_age < .5:
        raise RuntimeError('Stale SDK LowState (<100ms) or RobotState (<500ms)')


def require_low_hold(environment, mode, state, low_age, mode_age):
    require_state(environment, low_age, mode_age)
    if mode != 3:
        raise RuntimeError('Confirmed LOW ownership required during hold')
    if state is None or any(np.asarray(state[key]).shape != (31,) or not np.isfinite(state[key]).all()
                            for key in ('q', 'dq', 'status')) or any(state['status']):
        raise RuntimeError('Fresh finite error-free all31 state required during LOW hold')


def prepare_serial_pd_message(message, record, serial_pd, environment, mode, state,
                              state_at, mode_at, clock, stage):
    """Validate and prepare an immediately published PD record without callbacks.

    The outer loop drains ROS callbacks before sampling native coordinates. No
    callback work may age that sample between PD calculation and publication.
    Ownership, SDK state, and both publication clocks still gate every record.
    """
    from htd_serial_pd import write_motor_fields
    now = time.monotonic()
    require_low_hold(environment, mode, state, now-state_at, now-mode_at)
    stage['message_start_monotonic'] = time.monotonic()
    write_motor_fields(message, record)
    stage['message_done_monotonic'] = time.monotonic()
    publish_clock = clock()
    publish_monotonic = time.monotonic()
    require_state(environment, publish_monotonic-state_at, publish_monotonic-mode_at)
    serial_pd.check_publication(record, native_time=publish_clock['sim'], wall_time=publish_monotonic)
    stage['publication_check_monotonic'] = publish_monotonic
    return publish_clock, publish_monotonic


def slow_start_records(log_text):
    """Only report observed values; missing inactive logs are not proof of expiry.

    The supplied binary returns before its logging block once inactive. Keep
    this evidence separate from the wall-clock hold duration.
    """
    pattern = (r'SlowStartModule elapsedSec: ([\d.eE+-]+), slowStartActive: (true|false), '
               r'maxQDelta: ([\d.eE+-]+), maxQDeltaJoint: (\w+), errorJoints: \[([^\]]*)\]')
    return [dict(elapsed_s=float(elapsed), active=active == 'true', max_q_delta_rad=float(delta),
                 max_q_delta_joint=joint, error_joints=errors)
            for elapsed, active, delta, joint, errors in re.findall(pattern, log_text)]


def runtime_recovered(log_text):
    """The last degraded controller state must have a later recovery record."""
    degraded = log_text.rfind('Runtime SHM degraded;')
    recovered = log_text.rfind('Runtime SHM recovered;')
    return recovered >= 0 and recovered > degraded


def velocity_command(elapsed, previous, dt, stand_seconds, move_seconds, desired):
    target = np.asarray(desired, dtype=float) if stand_seconds <= elapsed < stand_seconds+move_seconds else np.zeros(3)
    limit = np.array([.15, .15, .5])*dt
    return np.asarray(previous)+np.clip(target-np.asarray(previous), -limit, limit)


class PublishedCommand:
    """Cleanup uses only commands acknowledged by the ROS publisher call."""
    def __init__(self, q, dq):
        self.remember(q, dq)

    def remember(self, q, dq):
        self.q, self.dq = np.asarray(q).copy(), np.asarray(dq).copy()

    def restore(self):
        return self.q.copy(), self.dq.copy()


def request_pd_pacing(out, rate, clock):
    """Request existing native wall pacing once; log request, not an invented ack."""
    if rate is None:
        return None
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError('Simulation wall-rate limit must be finite and within (0,1]')
    from run import request_replay_pacing
    before = clock()
    request_monotonic = time.monotonic()
    if not request_replay_pacing(out, rate, 0., 0.):
        raise FileExistsError('PD pacing must be requested once after this reset')
    after = clock()
    return dict(label='serial_pd_pacing_requested', rate=rate, wall=time.time(),
                request_monotonic=request_monotonic, request_file_mtime_ns=(out/'pacing.request').stat().st_mtime_ns,
                clock_before=before, clock_after=after, native_activation_acknowledged=False,
                activation_contract='Existing pacer checks request before the next native mj_step; effect audited from native/wall timestamps')


def worker(out, args):
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Worker requires the scoped private simulation container')
    runtime = json.loads((out/'runtime.json').read_text())
    configuration = yaml.safe_load((out/'robot.yaml').read_text())['igris_c']
    if runtime['network'] != 'none' or runtime['ipc'] != 'private' or runtime.get('hardware_executed') is not False:
        raise RuntimeError('Private simulation runtime required')
    if runtime.get('simulation_rate_limit') != args.simulation_rate_limit:
        raise RuntimeError('Native pacing runtime differs from the explicit probe option')
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest.get('htd_backlash_limit_timeconst_s') != args.backlash_limit_timeconst:
        raise RuntimeError('Backlash numerical variant differs from the explicit probe option')
    backlash_variant = None
    if args.backlash_limit_timeconst is not None:
        backlash_variant = json.loads((out/'htd_backlash_limit_audit.json').read_text())
        if (not backlash_variant['pass'] or backlash_variant['timeconst_s'] != args.backlash_limit_timeconst
                or backlash_variant['variant_scene_sha256'] != hashlib.sha256((out/'scene.xml').read_bytes()).hexdigest()):
            raise RuntimeError('Fresh-scene backlash numerical variant audit/hash mismatch')
    if configuration['controller']['GravityCompensationModule']['enabled'] is not False:
        raise RuntimeError('Gravity compensation must be disabled before container launch')
    if Path(configuration['description']['mujoco']['robot_model_path']) != out/'scene.xml':
        raise RuntimeError('Unexpected simulator scene')
    if (out/'room_layout.request').exists() or (out/'room_layout_applied.json').exists():
        raise RuntimeError('Empty-hand probe requires the room to remain parked')
    import torch
    os.sched_setaffinity(0, {args.worker_cpu})
    torch.set_num_threads(1)
    artifacts = probe_artifacts(args)
    require_artifact_manifest(artifacts, json.loads((out/'manifest.json').read_text()))
    from htd_initial_state import validate as validate_initial_state
    initial_state = validate_initial_state(out)
    require_initial_state(artifacts, initial_state)
    policy = HTDPolicy(artifacts.policy_path, artifacts.deployment_path)
    baseline_q = artifacts.hold_q
    compensation_indices = POLICY_SDK_INDICES if args.output_policy_joints_only or args.serial_pd_torque else None
    compensation_mask = np.ones(31, dtype=bool)
    if compensation_indices is not None:
        compensation_mask[:] = False
        compensation_mask[list(compensation_indices)] = True
    kp, kd, gains = load_gains(map_waist=args.map_waist_gains, training_source=artifacts.training_source)
    (out/'htd_gains.json').write_text(json.dumps(gains, indent=2)+'\n')
    mount, gate = imu_mount(out/'vendor.xml'), TargetLimits()
    gate.check(baseline_q)
    serial_pd = None
    if args.serial_pd_torque:
        from htd_serial_pd import SerialPD, zero_owned_effort, warm_policy_actor
        serial_pd = SerialPD(baseline_q, kp, kd, gate, delay_steps=args.serial_pd_delay_steps,
                             training_source=artifacts.training_source)
        warm_policy_actor(policy)
        (out/'htd_serial_pd_contract.json').write_text(json.dumps(serial_pd.contract, indent=2)+'\n')
    if args.project_targets:
        # Import/warm the bounded QP before any simulator services, so its first
        # use cannot consume a native policy deadline in the active loop.
        warm_target = baseline_q.copy()
        warm_target[1:3] = [.10, .25]
        gate.project(warm_target)
    sys.path.insert(0, str(ROOT))
    from igris.hl_ik.native_clock import read_clock
    if args.output_coordinates:
        from htd_output_coordinates import read_output, compensate_output, correction_evidence, FIELDS, MAX_BIAS_VELOCITY
    def coordinate_sample():
        return read_output(out/'output_coordinates.bin', native_time=lambda: read_clock(out/'native_clock.bin')['sim'])
    from control_gc import defer_cyclic_gc
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd, LowState, RobotState, BmsState
    from igris_c_sdk.srv import ControlModeCommandRequest, MujocoSimCmd, TorqueCmd
    rclpy.init()
    node = rclpy.create_node('htd_native_empty_hand_probe')
    state = None
    mode = None
    environment = None
    state_at = mode_at = 0.
    bms = None
    bms_at = 0.
    bms_header = None
    count = 0
    last_publish = 0.
    target = baseline_q.copy()
    target_velocity = np.zeros(31)
    published_command = PublishedCommand(target, target_velocity)
    coordinate_active = False
    serial_pd_active = False
    pd_stage = {}
    projected = target.copy()
    raw_target = target.copy()
    events, samples, command_samples, pd_samples = [], [], [], []
    summary = dict(hardware_executed=False, loaded=False, pass_=False, error=None,
                   policy_sha256=policy.policy_sha256, device='cpu', torch_version=torch.__version__,
                   artifacts=artifacts.evidence, initial_state=initial_state,
                   imu_mount_quaternion_torso_to_sensor=mount.tolist(),
                   observation_coordinates=('Native serial hinge plus backlash q/dq; same-callback vendor FK encoder estimate'
                                            if args.output_coordinates else 'SDK joint_state serial q/dq, no output-angle/backlash correction'),
                   output_coordinates=args.output_coordinates,
                   saturate_output_velocity=args.saturate_output_velocity,
                   output_policy_joints_only=args.output_policy_joints_only,
                   project_encoder_targets=args.project_encoder_targets,
                   map_waist_gains=args.map_waist_gains,
                   serial_pd_torque=args.serial_pd_torque,
                   simulation_rate_limit=args.simulation_rate_limit,
                   backlash_limit_variant=backlash_variant,
                   compensation_sdk_indices=np.flatnonzero(compensation_mask).tolist() if args.output_coordinates else [],
                   hands='Existing native MCP callback default30deg; no external hand controller, no payload held',
                   room_parked=True, stand_seconds=args.stand_seconds, move_seconds=args.move_seconds,
                   stop_seconds=args.stop_seconds if args.move_seconds else 0., command=args.command,
                   project_targets=args.project_targets, projection_limit_rad=args.projection_budget_rad if args.project_targets else None,
                   correction_budgets=dict(aggregate_target_projection_rad=args.projection_budget_rad,
                       encoder_projection_rad=.10, owned_position_bias_rad=.10,
                       raw_to_command_rad=args.raw_command_budget_rad, velocity_compensation_rad_s=2.),
                   low_warmup_seconds=args.low_warmup_seconds,
                   events=events, startup='stream default PJS31, resume, torque, HOME, LOW, optional wall-time hold, one SIM reset, immediately HTD')
    if serial_pd is not None:
        summary['correction_budgets'] = dict(owned_position_bias_rad=.10,
            target_projection='Not used: raw reference drives effort, published q is feasible default',
            velocity_compensation='Not used: physical output damping is included in serial effort')
    def on_mode(message):
        nonlocal mode, environment, mode_at
        if mode != int(message.state):
            events.append(dict(label='robot_state_transition', state=int(message.state), environment=int(message.environment), wall=time.time()))
            print('MODE', events[-1], flush=True)
        mode, environment, mode_at = int(message.state), int(message.environment), time.monotonic()
        if environment != RobotState.ROBOT_ENV_SIM:
            raise RuntimeError('Received non-SIM RobotState')
    def on_state(message):
        nonlocal state, state_at
        state = dict(q=np.array([j.q for j in message.joint_state]), dq=np.array([j.dq for j in message.joint_state]),
                     quat=np.array(message.imu_state.quaternion), gyro=np.array(message.imu_state.gyroscope),
                     status=[int(j.status_bits) for j in message.joint_state])
        state_at = time.monotonic()
    def on_bms(message):
        nonlocal bms, bms_at, bms_header
        status = {key: int(getattr(message, key)) for key in
                  ('bms_init_state', 'body_power', 'legs_power', 'estop')}
        bms_at = time.monotonic()
        bms_header = {key: int(getattr(message.header, key)) for key in ('seq', 'sec', 'nanosec')}
        if status != bms:
            events.append(dict(label='bms_state_transition', wall=time.time(), monotonic=bms_at,
                               status=status, header=bms_header))
            print('BMS', events[-1], flush=True)
        bms = status
    node.create_subscription(LowState, 'igris_c_001/rt/lowstate', on_state, qos_profile_sensor_data)
    node.create_subscription(RobotState, 'igris_c_001/rt/robotstate', on_mode, qos_profile_sensor_data)
    node.create_subscription(BmsState, 'igris_c_001/rt/bmsstate', on_bms, qos_profile_sensor_data)
    publisher = node.create_publisher(LowCmd, 'igris_c_001/rt/lowcmd', qos_profile_sensor_data)
    message = LowCmd()
    message.kinematic_modes = [1]*5
    for index, motor in enumerate(message.motors):
        motor.id = index
        motor.dq = motor.tau = 0.
        motor.kp, motor.kd = float(kp[index]), float(kd[index])
    def pump(force=False, pd_record=None):
        nonlocal count, last_publish, target, target_velocity
        if pd_record is not None:
            pd_stage['pump_enter_monotonic'] = time.monotonic()
        else:
            # All ROS callback work precedes the latest output-coordinate sample.
            # A computed PD record takes the guarded immediate publication path.
            for _ in range(6):
                rclpy.spin_once(node, timeout_sec=0)
        now = time.monotonic()
        if pd_record is not None:
            pd_stage['pump_spin_done_monotonic'] = now
        if environment is not None:
            if environment != RobotState.ROBOT_ENV_SIM or now-mode_at >= .5:
                raise RuntimeError('Fresh SIM state required for publication')
            # Once torque control is active there is no 100 Hz wall-time resend.
            # Only an independently due native 5 ms PD record may be published.
            publish_due = pd_record is not None if serial_pd_active else (force or now-last_publish >= .01)
            if publish_due:
                coordinate = None
                if coordinate_active:
                    coordinate = coordinate_sample()
                    try:
                        target, target_velocity, bias, dbias = compensate_output(
                            raw_target, projected, coordinate, gate, saturate_velocity=args.saturate_output_velocity,
                            owned_indices=compensation_indices, project_encoder=args.project_encoder_targets,
                            projection_budget_rad=args.projection_budget_rad, raw_command_budget_rad=args.raw_command_budget_rad)
                    except ValueError as exc:
                        summary['rejected_output_command'] = dict(
                            error=str(exc), sim=coordinate['sim'], sequence=coordinate['sequence'],
                            wall=time.time(), native_coordinates={k: coordinate[k].tolist() for k in FIELDS},
                            raw_target=raw_target.tolist(), projected_target=projected.tolist())
                        raise
                stamp = time.time_ns()
                message.header.seq = count
                message.header.sec, message.header.nanosec = divmod(stamp, 1_000_000_000)
                if pd_record is not None:
                    publish_clock, publish_monotonic = prepare_serial_pd_message(
                        message, pd_record, serial_pd, environment, mode, state, state_at, mode_at,
                        lambda: read_clock(out/'native_clock.bin'), pd_stage)
                else:
                    for index, motor in enumerate(message.motors):
                        motor.q = float(target[index])
                        motor.dq = float(target_velocity[index])
                publisher.publish(message)
                if pd_record is not None:
                    publish_end = time.monotonic()
                    pd_row = {field.name: getattr(pd_record, field.name) for field in fields(pd_record)}
                    pd_row.update(pd_stage)
                    pd_row.update(publication_native_time=publish_clock['sim'], publication_monotonic=publish_monotonic,
                                  publication_wall=stamp*1e-9, publication_lag_s=publish_clock['sim']-pd_record.scheduled_time,
                                  sdk_state_age_s=publish_monotonic-state_at, publish_call_wall_s=publish_end-publish_monotonic,
                                  callbacks_after_pd=0,
                                  sdk_q=state['q'].copy(), sdk_dq=state['dq'].copy(),
                                  post_publish_native_time=float('nan'))
                    pd_samples.append(pd_row)
                published_command.remember(target, target_velocity)
                if coordinate is not None:
                    command_record = dict(wall=stamp*1e-9, sim=coordinate['sim'], sequence=coordinate['sequence'],
                                                q=target.copy(), dq=target_velocity.copy(), bias=bias.copy(), dbias=dbias.copy(),
                                                dq_raw=np.where(compensation_mask, -dbias, 0.),
                                                dq_saturated=compensation_mask & (np.abs(target_velocity+dbias)>1e-12),
                                                projected=projected.copy(), raw_target=raw_target.copy(),
                                                observer_wall_age=coordinate['wall_age'], observer_native_lag=coordinate['native_lag'])
                    command_record.update(correction_evidence(raw_target, projected, target, bias, compensation_mask))
                    command_samples.append(command_record)
                count += 1
                last_publish = now
                if pd_record is not None:
                    post_publish_clock = read_clock(out/'native_clock.bin')
                    pd_row['post_publish_native_time'] = post_publish_clock['sim']
                    serial_pd.check_publication(pd_record, native_time=post_publish_clock['sim'], wall_time=publish_end)
        if (out/'stop.request').exists():
            raise RuntimeError('Scoped stop requested')
    def wait_until(predicate, timeout, label):
        deadline = time.monotonic()+timeout
        while not predicate():
            pump()
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out: '+label)
            time.sleep(.001)
    clients = {}
    def discover_services():
        available = node.get_service_names_and_types()
        for key, cls in [('sim', MujocoSimCmd), ('torque', TorqueCmd), ('mode', ControlModeCommandRequest)]:
            typename = 'igris_c_sdk/srv/'+cls.__name__
            matches = [name for name, types in available if name.startswith('/igris_c_001/') and typename in types]
            if len(matches) > 1:
                raise RuntimeError('Ambiguous private service: '+typename)
            if matches and key not in clients:
                clients[key] = node.create_client(cls, matches[0])
        return len(clients) == 3 and all(client.service_is_ready() for client in clients.values())
    def call(key, command_type=None, preset='', label=None, timeout=5.):
        if environment != RobotState.ROBOT_ENV_SIM:
            raise RuntimeError('SIM service gate failed')
        client = clients[key]
        request = client.srv_type.Request()
        request.request_id = f'htd-probe-{time.time_ns()}'
        if key == 'torque':
            request.torque = command_type
        else:
            request.command_type = command_type
        if key == 'mode':
            request.preset_id, request.is_cyclic = preset, False
        future = client.call_async(request)
        wait_until(future.done, timeout, label or key)
        response = future.result()
        events.append(dict(label=label or key, success=bool(response.success), message=response.message, wall=time.time()))
        print('SERVICE', events[-1], flush=True)
        if not response.success:
            raise RuntimeError('Service rejected: '+str(events[-1]))
    started = time.monotonic()
    pd_record = None
    try:
        wait_until(lambda: environment == RobotState.ROBOT_ENV_SIM, 10., 'SIM RobotState')
        wait_until(discover_services, 10., 'ROS simulation services')
        until = time.monotonic()+.25
        wait_until(lambda: time.monotonic() >= until, 1., 'prestream')
        call('sim', 1, label='resume')
        wait_until(lambda: state is not None and (out/'physics_stats.json').exists(), 5., 'native state')
        stats = json.loads((out/'physics_stats.json').read_text())
        process = Path('/proc')/str(stats['pid'])
        tid = stats['tid']
        if (process/'comm').read_text().strip() != 'igris_c_mujoco' or not (process/'task'/str(tid)).exists():
            raise RuntimeError('Physics thread ownership changed')
        os.sched_setaffinity(tid, {args.physics_cpu})
        subprocess.run(['chrt', '-f', '-p', '50', str(tid)], check=True)
        summary['physics_scheduling'] = dict(cpu=args.physics_cpu, fifo_priority=50, pid=stats['pid'], tid=tid)
        summary['worker_cpu'] = args.worker_cpu
        call('torque', 1, label='torque_on')
        torque_ack_at = time.monotonic()
        wait_until(lambda: runtime_recovered((out/'vendor.log').read_text(errors='replace'))
                   and state is not None and time.monotonic()-state_at < .1 and not any(state['status']),
                   5., 'controller SHM recovery after torque-on')
        wait_until(lambda: runtime_recovered((out/'vendor.log').read_text(errors='replace'))
                   and state is not None and time.monotonic()-state_at < .1 and not any(state['status'])
                   and torque_ready(bms, received_at=bms_at, torque_ack_at=torque_ack_at,
                                    now=time.monotonic()),
                   5., 'fresh BMS torque-ready state after torque-on')
        event = dict(label='torque_ready_verified', wall=time.time(), torque_ack_monotonic=torque_ack_at,
                     bms_received_monotonic=bms_at, bms_status=bms, bms_header=bms_header,
                     wait_after_ack_wall_s=time.monotonic()-torque_ack_at)
        events.append(event)
        print('TORQUE_READY', json.dumps(event), flush=True)
        # The native recovery transition can replace a previously accepted HOME
        # with JOINT_HOLD. Wait for that transition first; the old native WBC
        # launcher also used HIGH_LEVEL_JOINT_CONTROL + HOME as one fallback.
        for attempt in range(2):
            if attempt:
                call('mode', 8, label='high_joint_recovery')
            call('mode', 0, preset='HOME', label=f'home_{attempt+1}')
            try:
                wait_until(lambda: mode == RobotState.ROBOT_STATE_HIGH_MOTION_STANDBY, 10., 'HOME standby')
                break
            except RuntimeError as exc:
                if attempt or 'Timed out: HOME standby' not in str(exc):
                    raise
        call('mode', 5, label='low')
        wait_until(lambda: mode == RobotState.ROBOT_STATE_LOW, 3., 'LOW state')
        if args.low_warmup_seconds:
            # Empty hands only. The unsupported pre-reset HOME/LOW pose may
            # fall while the valid hold stream runs; this is not actor evidence.
            # Do not add such a reset to any loaded transport controller.
            warmup_started = time.monotonic()
            warmup_publishes = count
            for label in ('low_warmup_start', 'low_warmup_end'):
                if label.endswith('end'):
                    while time.monotonic()-warmup_started < args.low_warmup_seconds:
                        pump()
                        now = time.monotonic()
                        require_low_hold(environment, mode, state, now-state_at, now-mode_at)
                        time.sleep(.001)
                event = dict(label=label, wall=time.time(), sim=read_clock(out/'native_clock.bin')['sim'],
                             elapsed_wall_s=time.monotonic()-warmup_started,
                             default_hold_commands_sent=count-warmup_publishes,
                             slow_start_records=slow_start_records((out/'vendor.log').read_text(errors='replace')),
                             direct_inactive_transition_observed=False)
                events.append(event)
                print('WARMUP', json.dumps(event), flush=True)
        pre_reset_log_size = (out/'vendor.log').stat().st_size
        old_clock = read_clock(out/'native_clock.bin')['sim']
        reset_wall = time.time()
        if initial_state is not None:
            from htd_initial_state import request as request_initial_state
            request_initial_state(out)
            events.append(dict(label='initial_state_request', wall=reset_wall, sim=old_clock))
        call('sim', 3, label='policy_reset')
        # Drain queued SDK samples and explicitly see the reset clock before
        # pinning the policy origin; do not infer using pre-reset history.
        reset_received = time.monotonic()
        wait_until(lambda: state_at > reset_received and read_clock(out/'native_clock.bin')['sim'] < old_clock,
                   2., 'fresh post-reset SDK state')
        def fresh_reset_pose():
            raw = np.frombuffer((out/'latest.bin').read_bytes(), dtype=np.float64)
            return len(raw) > 2 and raw[0] > reset_wall and 0 <= raw[1] < old_clock
        wait_until(fresh_reset_pose, 2., 'fresh post-reset native pose')
        if initial_state is not None:
            from htd_initial_state import acknowledgement
            def initial_state_applied():
                try:
                    ack = acknowledgement(out)
                except FileNotFoundError:
                    return False
                if (out/'htd_initial_state_applied.json').stat().st_mtime < reset_wall:
                    raise RuntimeError('Initial-state acknowledgement predates this reset')
                summary['initial_state_acknowledgement'] = ack
                return True
            wait_until(initial_state_applied, 2., 'fresh explicit initial-state acknowledgement')
            events.append(dict(label='initial_state_verified', wall=time.time(),
                               acknowledgement=summary['initial_state_acknowledgement']))
        if args.output_coordinates:
            def fresh_reset_coordinates():
                try:
                    c = coordinate_sample()
                    return c['wall'] > reset_wall and c['sim'] < old_clock
                except (RuntimeError, OSError):
                    return False
            wait_until(fresh_reset_coordinates, 2., 'fresh post-reset output coordinates')
            summary['output_coordinate_contract'] = dict(
                observer=json.loads((out/'htd_output_observer.json').read_text()),
                velocity_correction_guard_rad_s=MAX_BIAS_VELOCITY,
                total_raw_to_command_guard_rad=args.raw_command_budget_rad,
                aggregate_target_projection_guard_rad=args.projection_budget_rad,
                encoder_projection_guard_rad=.10,
                owned_position_bias_guard_rad=.10,
                exact_scalar_pd_algebra_only_when_unsaturated=True,
                velocity_saturation_enabled=args.saturate_output_velocity,
                encoder_projection_enabled=args.project_encoder_targets,
                parallel_actuator_parity=False,
                limitation='Parallel nonlinear IK, diagonal motor gains and native delay remain; this tests bounded coordinate compensation.')
            if serial_pd is not None:
                summary['output_coordinate_contract'] = dict(
                    observer=json.loads((out/'htd_output_observer.json').read_text()),
                    use='True output q/dq for original observation and independently scheduled serial PD',
                    owned_position_bias_guard_rad=.10, owned_actual_q_polytope_checked=True,
                    coordinate_target_compensation=False, parallel_actuator_parity=False)
        if args.simulation_rate_limit is not None:
            now = time.monotonic()
            require_low_hold(environment, mode, state, now-state_at, now-mode_at)
            event = request_pd_pacing(out, args.simulation_rate_limit, lambda: read_clock(out/'native_clock.bin'))
            events.append(event)
            print('PACING', json.dumps(event), flush=True)
        if serial_pd is None:
            policy.reset()
        events.append(dict(label='policy_history_reset', wall=time.time(), previous_raw_action_zero=True,
                           phase_clock_zero=True, policy_samples=len(samples)))
        origin = read_clock(out/'native_clock.bin')['sim']
        serial_pd_pending = serial_pd is not None
        if serial_pd is not None:
            serial_pd_active = True
            summary['serial_pd_contract'] = serial_pd.contract
        total = args.stand_seconds+args.move_seconds+(args.stop_seconds if args.move_seconds else 0.)
        command = np.zeros(3)
        model = json.loads((out/'model.json').read_text())
        names = [body['name'] for body in model['bodies']]
        pelvis, torso = names.index('pelvis'), names.index('torso')
        next_status = 0.
        active_wall = time.monotonic()
        with defer_cyclic_gc():
            while True:
                pd_record = None
                if serial_pd is not None:
                    pd_stage = dict.fromkeys(('loop_monotonic', 'callbacks_start_monotonic', 'callbacks_done_monotonic',
                        'initial_output_read_start_monotonic', 'initial_output_read_done_monotonic',
                        'initial_due_check_monotonic', 'initial_due_check_native', 'actor_start_monotonic', 'actor_done_monotonic',
                        'output_read_start_monotonic', 'output_read_done_monotonic',
                        'pd_compute_start_monotonic', 'pd_compute_done_monotonic',
                        'pump_enter_monotonic', 'pump_spin_done_monotonic', 'message_start_monotonic',
                        'message_done_monotonic', 'publication_check_monotonic'), float('nan'))
                    pd_stage['loop_monotonic'] = time.monotonic()
                    pd_stage['callbacks_start_monotonic'] = time.monotonic()
                pump()
                now = time.monotonic()
                if serial_pd is not None:
                    pd_stage['callbacks_done_monotonic'] = now
                require_state(environment, now-state_at, now-mode_at)
                if mode != RobotState.ROBOT_STATE_LOW:
                    raise RuntimeError('LOW control ownership lost')
                if any(state['status']):
                    raise RuntimeError('SDK joint status error: '+str(state['status']))
                clock = read_clock(out/'native_clock.bin')
                elapsed = clock['sim']-origin
                raw = np.frombuffer((out/'latest.bin').read_bytes(), dtype=np.float64)
                if raw.shape != (2+7*len(names),) or not np.isfinite(raw).all() or not 0 <= time.time()-raw[0] < .5:
                    raise RuntimeError('Stale/invalid 30Hz native pose observer')
                poses = raw[2:].reshape(-1, 7)
                w, x, y, z = poses[pelvis, 3:]
                tilt = math.acos(float(np.clip(1-2*(x*x+y*y), -1, 1)))
                if poses[pelvis, 2] < .7 or tilt > .4:
                    raise RuntimeError(f'Native fall guard: pelvis_z={poses[pelvis, 2]:.4f}, tilt={tilt:.4f}')
                if elapsed >= total:
                    summary['pass_'] = True
                    break
                if now-active_wall > args.wall_timeout:
                    raise RuntimeError('Native probe wall timeout')
                if serial_pd is not None:
                    pd_stage['initial_output_read_start_monotonic'] = time.monotonic()
                coordinates = coordinate_sample() if args.output_coordinates else None
                if serial_pd is not None:
                    pd_stage['initial_output_read_done_monotonic'] = time.monotonic()
                observation_time = coordinates['sim'] if coordinates is not None else clock['sim']
                if serial_pd is not None:
                    if serial_pd_pending:
                        # Anchor after setup/pose validation, directly on the
                        # first fresh output sample; no startup deadline debt.
                        origin = observation_time
                        elapsed = clock['sim']-origin
                        serial_pd.reset(origin)
                        policy.reset(sim_time=origin)
                        serial_pd_pending = False
                        events.append(dict(label='serial_pd_activated', wall=time.time(), sim=origin,
                            actor_period_s=policy.dt, pd_period_s=serial_pd.dt,
                            initial_raw_history_zero=True, initial_delay_history='repeat first raw reference'))
                    # Guard a missed PD interval even when the actor is not due.
                    pd_stage['initial_due_check_native'] = read_clock(out/'native_clock.bin')['sim']
                    pd_stage['initial_due_check_monotonic'] = time.monotonic()
                    serial_pd.due(coordinates, native_time=pd_stage['initial_due_check_native'],
                                  wall_time=pd_stage['initial_due_check_monotonic'])
                if policy.policy_due(observation_time):
                    command = velocity_command(elapsed, command, policy.dt, args.stand_seconds, args.move_seconds, args.command)
                    quat, gyro = torso_imu(state['quat'], state['gyro'], mount)
                    obs_q, obs_dq = (coordinates['q_output'], coordinates['dq_output']) if coordinates is not None else (state['q'], state['dq'])
                    if serial_pd is not None:
                        pd_stage['actor_start_monotonic'] = time.monotonic()
                    result = policy.step(obs_q, obs_dq, quat, gyro, command, observation_time)
                    if serial_pd is not None:
                        pd_stage['actor_done_monotonic'] = time.monotonic()
                    raw_target = artifacts.merge_actor_target(result.q_target)
                    try:
                        if serial_pd is not None:
                            # This raw reference is never published as q. Only
                            # the independent serial PD consumes it; no clipping.
                            projected = raw_target.copy()
                            target, target_velocity = baseline_q.copy(), np.zeros(31)
                            bias = coordinates['q_output']-coordinates['q_encoder']
                            dbias = coordinates['dq_output']-coordinates['dq_encoder']
                        else:
                            projected = gate.project(raw_target, max_correction_rad=args.projection_budget_rad) if args.project_targets else raw_target.copy()
                            gate.check(projected)
                        if coordinates is not None and serial_pd is None:
                            target, target_velocity, bias, dbias = compensate_output(
                                raw_target, projected, coordinates, gate, saturate_velocity=args.saturate_output_velocity,
                                owned_indices=compensation_indices, project_encoder=args.project_encoder_targets,
                                projection_budget_rad=args.projection_budget_rad, raw_command_budget_rad=args.raw_command_budget_rad)
                            coordinate_active = True
                        elif serial_pd is None:
                            target = projected.copy()
                    except ValueError:
                        summary['rejected_policy_step'] = dict(sim=clock['sim'], q_target=result.q_target.tolist(),
                                                               raw_action=result.raw_action.tolist(), observation=result.observation.tolist())
                        if coordinates is not None:
                            summary['rejected_policy_step']['native_coordinates'] = {k: coordinates[k].tolist() for k in FIELDS}
                        raise
                    # Offsets are evidence only, not fed into the observation.
                    offsets = np.full(31, np.nan)
                    with (out/'joint_offsets.bin').open('rb') as stream:
                        size = stream.seek(0, 2)//256
                        if size:
                            stream.seek((size-1)*256)
                            offset_row = np.frombuffer(stream.read(256), dtype=np.float64)
                            if len(offset_row) == 32:
                                offsets = offset_row[1:]
                    record = dict(wall=time.time(), sim=observation_time, elapsed=elapsed,
                                        policy_time=result.policy_time, lag_s=result.lag_s,
                                        state_age_s=now-state_at, pose_sim=float(raw[1]), pelvis=poses[pelvis].copy(), torso=poses[torso].copy(),
                                        q=state['q'].copy(), dq=state['dq'].copy(), target=target.copy(),
                                        target_velocity=target_velocity.copy(), target_projected=projected.copy(),
                                        target_unprojected=raw_target.copy(), actor_full_target=result.q_target.copy(),
                                        projection_max_rad=float(np.max(np.abs(projected-raw_target))),
                                        obs=result.observation, action=result.raw_action, command=command.copy(),
                                        quat=quat.copy(), gyro=gyro.copy(), joint_offsets=offsets.copy())
                    if coordinates is not None:
                        record.update({name: coordinates[name].copy() for name in FIELDS})
                        if serial_pd is None:
                            record.update(correction_evidence(raw_target, projected, target, bias, compensation_mask))
                        record.update(coordinate_sequence=coordinates['sequence'], coordinate_bias=bias.copy(),
                                      coordinate_dbias=dbias.copy(), fk_residual=coordinates['q_hinge']-coordinates['q_encoder'],
                                      target_velocity_raw=np.zeros(31) if serial_pd is not None else np.where(compensation_mask, -dbias, 0.),
                                      target_velocity_saturated=np.zeros(31, dtype=bool) if serial_pd is not None else compensation_mask & (np.abs(target_velocity+dbias)>1e-12),
                                      fk_velocity_residual=coordinates['dq_hinge']-coordinates['dq_encoder'],
                                      observer_wall_age=coordinates['wall_age'], observer_native_lag=coordinates['native_lag'])
                    samples.append(record)
                    if serial_pd is None:
                        pump(force=True)
                if serial_pd is not None:
                    # Resample after inference/file work. Never reuse an old
                    # state or catch up by replaying missed 5 ms PD ticks.
                    pd_stage['output_read_start_monotonic'] = time.monotonic()
                    coordinates = coordinate_sample()
                    pd_stage['output_read_done_monotonic'] = time.monotonic()
                    pd_now = read_clock(out/'native_clock.bin')['sim']
                    if serial_pd.due(coordinates, native_time=pd_now, wall_time=time.monotonic()):
                        pd_stage['pd_compute_start_monotonic'] = time.monotonic()
                        pd_record = serial_pd.step(raw_target, coordinates, native_time=pd_now, wall_time=time.monotonic())
                        pd_stage['pd_compute_done_monotonic'] = time.monotonic()
                        pump(pd_record=pd_record)
                if elapsed >= next_status:
                    print(f'HTD {elapsed:.3f}s, pelvis_z={poses[pelvis, 2]:.4f}, tilt={tilt:.4f}, samples={len(samples)}', flush=True)
                    next_status += 1.
                time.sleep(.0002 if serial_pd is not None else .001)
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        summary['traceback'] = traceback.format_exc()
        print(summary['traceback'], flush=True)
        if serial_pd is not None:
            summary['serial_pd_guard'] = dict(error=str(exc), next_due=serial_pd.next_due,
                                             pd_computations=serial_pd.step_count, pd_publications=len(pd_samples),
                                             stale_poll_waits=serial_pd.stale_poll_waits,
                                             deadline_guard_triggered='deadline' in str(exc).lower(),
                                             raw_target=raw_target.tolist(), stage_monotonic=pd_stage)
            if 'coordinates' in locals() and coordinates is not None:
                summary['serial_pd_guard']['output_sample'] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                               for key, value in coordinates.items()}
            if pd_record is not None:
                summary['serial_pd_guard']['candidate'] = {
                    field.name: (getattr(pd_record, field.name).tolist() if isinstance(getattr(pd_record, field.name), np.ndarray)
                                 else getattr(pd_record, field.name)) for field in fields(pd_record)}
    finally:
        coordinate_active = False
        serial_pd_active = False
        # pause_finally spins/publishes while waiting for its response. Restore
        # the last SENT command so a rejected, unsent candidate cannot leak out.
        target, target_velocity = published_command.restore()
        if serial_pd is not None:
            # Zero held feedforward before the pause service spins/publishes;
            # keep upper16 baseline gains and never reactivate owned motor PD.
            zero_owned_effort(message)
            target, target_velocity = baseline_q.copy(), np.zeros(31)
            summary['serial_pd_cleanup'] = 'Owned15 tau/kp/kd zero; feasible default q; upper16 baseline retained'
        try:
            if 'sim' in clients and environment == RobotState.ROBOT_ENV_SIM:
                call('sim', 0, label='pause_finally', timeout=2.)
        except Exception as exc:
            summary['pause_error'] = str(exc)
        summary.update(commands_sent=count, coordinate_commands_sent=len(command_samples),
                       policy_samples=len(samples), wall_seconds=time.monotonic()-started)
        vendor_bytes = (out/'vendor.log').read_bytes()
        summary['slow_start_evidence'] = dict(
            all_records=slow_start_records(vendor_bytes.decode(errors='replace')),
            post_reset_records=slow_start_records(vendor_bytes[pre_reset_log_size:].decode(errors='replace'))
                               if 'pre_reset_log_size' in locals() else [],
            direct_inactive_transition_observed=False,
            limitation='Vendor binary logs active hold only; inactive transition bypasses logging. Silence alone is not direct expiry proof.')
        if samples:
            arrays = {name: np.asarray([sample[name] for sample in samples]) for name in samples[0]}
            np.savez_compressed(out/'htd_samples.npz', **arrays)
            summary.update(completed_native_seconds=float(samples[-1]['elapsed']),
                           pelvis_z_min_m=float(arrays['pelvis'][:, 2].min()),
                           lag_max_s=float(arrays['lag_s'].max()), lag_p95_s=float(np.percentile(arrays['lag_s'], 95)),
                           sdk_state_age_max_s=float(arrays['state_age_s'].max()),
                           sdk_to_output_backlash_max_rad=float(np.nanmax(np.abs(arrays['joint_offsets']))),
                           target_projection_max_rad=float(arrays['projection_max_rad'].max()),
                           target_projection_affected_ticks=int(np.sum(arrays['projection_max_rad'] > 1e-10)))
            if args.output_coordinates:
                summary.update(coordinate_bias_max_rad=float(np.max(np.abs(arrays['coordinate_bias']))),
                               coordinate_dbias_max_rad_s=float(np.max(np.abs(arrays['coordinate_dbias']))),
                               native_fk_residual_max_rad=float(np.max(np.abs(arrays['fk_residual']))),
                               native_fk_velocity_residual_max_rad_s=float(np.max(np.abs(arrays['fk_velocity_residual']))),
                               observer_wall_age_max_s=float(arrays['observer_wall_age'].max()),
                               observer_native_lag_max_s=float(arrays['observer_native_lag'].max()))
        if command_samples:
            np.savez_compressed(out/'htd_output_commands.npz', **{k: np.asarray([s[k] for s in command_samples]) for k in command_samples[0]})
            masks = np.asarray([s['dq_saturated'] for s in command_samples])
            summary['velocity_compensation_saturation'] = dict(
                affected_publications=int(np.any(masks, axis=1).sum()),
                count_by_sdk_joint=masks.sum(axis=0).tolist(),
                raw_max_rad_s=float(np.max(np.abs([s['dq_raw'] for s in command_samples]))),
                applied_max_rad_s=float(np.max(np.abs([s['dq'] for s in command_samples]))),
                exact_output_pd_identity_lost=bool(masks.any()))
            corrections = np.asarray([s['encoder_projection'] for s in command_samples])
            summary['encoder_projection'] = dict(
                affected_publications=int(np.any(abs(corrections)>1e-12, axis=1).sum()),
                max_rad=float(np.max(abs(corrections))),
                aggregate_projection_budget_max_rad=float(max(s['projection_budget_used_rad'] for s in command_samples)),
                expected_output_residual_max_rad=float(np.max(np.abs([s['expected_output_target_residual'] for s in command_samples]))))
            summary['raw_to_command_max_rad'] = float(np.max(np.abs([s['q']-s['raw_target'] for s in command_samples])))
            summary['owned_position_bias_max_rad'] = float(np.max(np.abs([s['bias'][compensation_mask] for s in command_samples])))
        if pd_samples:
            pd_arrays = {name: np.asarray([s[name] for s in pd_samples]) for name in pd_samples[0]}
            np.savez_compressed(out/'htd_serial_pd_samples.npz', **pd_arrays)
            summary['serial_pd_timing'] = dict(
                computations=serial_pd.step_count, publications=len(pd_samples),
                stale_poll_waits=serial_pd.stale_poll_waits,
                scheduled_period_s=serial_pd.dt, native_deadline_lag_max_s=float(pd_arrays['publication_lag_s'].max()),
                output_native_lag_max_s=float(pd_arrays['native_lag_s'].max()),
                publish_call_max_wall_s=float(pd_arrays['publish_call_wall_s'].max()),
                measured_publication_intervals_native_s=np.diff(pd_arrays['publication_native_time']).tolist(),
                measured_publication_intervals_wall_s=np.diff(pd_arrays['publication_monotonic']).tolist(),
                training_effort_clipped_ticks=int(np.any(pd_arrays['training_clipped'], axis=1).sum()),
                native_serial_effort_clipped_ticks=int(np.any(pd_arrays['native_clipped'], axis=1).sum()),
                delivered_motor_effort_verified=False)
        summary['pass'] = summary.pop('pass_')
        (out/'htd_result.json').write_text(json.dumps(summary, indent=2)+'\n')
        node.destroy_node()
        rclpy.shutdown()
    return 0 if summary['pass'] else 1


def launch(out, args):
    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    artifacts = probe_artifacts(args)
    if artifacts.requires_initial_state and args.initial_state_file is None:
        raise ValueError('V2 native activation requires --initial-state-file')
    from box_replay import layout
    import run
    layout(out, compliant_hands=True, motion=args.motion, narrow_source_support=True, payload_mass=.25)
    trial = out/'run'
    configuration = yaml.safe_load((trial/'robot.yaml').read_text())
    configuration['igris_c']['controller']['GravityCompensationModule']['enabled'] = False
    (trial/'robot.yaml').write_text(yaml.safe_dump(configuration))
    from htd_backlash_limits import configure as configure_backlash_limits
    backlash_variant = configure_backlash_limits(trial, args.backlash_limit_timeconst)
    manifest = json.loads((trial/'manifest.json').read_text())
    manifest.update(htd_probe=True, empty_hand_only=True, gravity_compensation=False,
                    htd_artifacts=artifacts.evidence,
                    room_must_remain_parked=True, physics_cpu=args.physics_cpu, output_coordinates=args.output_coordinates,
                    saturate_output_velocity=args.saturate_output_velocity, output_policy_joints_only=args.output_policy_joints_only,
                    project_encoder_targets=args.project_encoder_targets,
                    map_waist_gains=args.map_waist_gains,
                    serial_pd_torque=args.serial_pd_torque, serial_pd_delay_steps=args.serial_pd_delay_steps,
                    projection_budget_rad=args.projection_budget_rad, raw_command_budget_rad=args.raw_command_budget_rad)
    if args.simulation_rate_limit is not None:
        manifest['simulation_rate_limit'] = args.simulation_rate_limit
    if backlash_variant is not None:
        manifest['htd_backlash_limit_timeconst_s'] = args.backlash_limit_timeconst
        manifest['htd_backlash_numerical_variant'] = backlash_variant
    (trial/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from htd_initial_state import prepare as prepare_initial_state
    initial_state = prepare_initial_state(trial, args.initial_state_file)
    require_initial_state(artifacts, initial_state)
    sources = [Path(__file__), HERE/'htd_policy.py', HERE/'run.py', HERE/'observer.cpp',
               ROOT/'igris/hl_ik/native_clock.cpp', ROOT/'igris/igris-c-wbc-base/scripts/slow_policy_runtime.py',
               trial/'vendor.xml', trial/'scene.xml', trial/'robot.yaml', HERE/'htd_probe_config.py',
               artifacts.training_source, artifacts.policy_path, artifacts.deployment_path,
               ROOT/'igris/configs/controller/params.yaml', ROOT/'igris/ws/install/igris_c_lib/lib/libigris_c_lib.so',
               ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco/igris_c_mujoco',
               ROOT/'igris/ws/install/igris_c_controller/lib/igris_c_controller/sim',
               ROOT/'igris/ws/install/igris_c_bridge/lib/igris_c_bridge/bridge_sim']
    if artifacts.native_hold_source is not None:
        sources += [artifacts.native_hold_source]
    if initial_state is not None:
        sources += [HERE/'htd_initial_state.py', HERE/'htd_initial_state.cpp', trial/'htd_initial_state.txt']
    if args.output_coordinates:
        sources += [HERE/'htd_output_coordinates.py', HERE/'htd_output_coordinates.cpp',
                    ROOT/'igris/ws/install/igris_c_shm/include/igris_c_shm/shm/state.h',
                    ROOT/'igris/ws/install/igris_c_shm/include/igris_c_shm/shm/common.h',
                    ROOT/'igris/ws/install/igris_c/include/igris_c/common/robot_define.h']
    if args.serial_pd_torque:
        sources += [HERE/'htd_serial_pd.py']
    if args.simulation_rate_limit is not None:
        sources += [HERE/'simulation_pacer.cpp']
    if backlash_variant is not None:
        sources += [HERE/'htd_backlash_limits.py', HERE/'htd_backlash_limits.cpp',
                    trial/'htd_backlash_original_scene.xml', trial/'htd_backlash_limit_audit.json']
    report = dict(hardware_executed=False, loaded=False, pass_=False, output=str(out), error=None,
                  artifacts=artifacts.evidence, initial_state=initial_state,
                  backlash_limit_variant=backlash_variant,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    try:
        environment = os.environ.copy()
        if args.output_coordinates:
            if environment.get('IGRIS_ROOM_PROFILE_LIBRARY'):
                raise RuntimeError('Refusing to replace another profile interposer')
            from htd_output_coordinates import build_observer
            observer = build_observer(trial, trial/'vendor.xml')
            environment['IGRIS_ROOM_PROFILE_LIBRARY'] = observer['library']
            report['output_observer'] = observer
        with (out/'start.log').open('w') as log:
            # start() retains its mandatory vendor/model audit before any launch.
            subprocess.run([sys.executable, str(HERE/'run.py'), 'start', '--out', str(trial), '--image', args.image],
                           stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
        argv = ['python3', str(Path(__file__).resolve()), 'worker', '--out', str(trial),
                '--stand-seconds', str(args.stand_seconds), '--move-seconds', str(args.move_seconds),
                '--stop-seconds', str(args.stop_seconds), '--command', *map(str, args.command),
                '--physics-cpu', str(args.physics_cpu), '--wall-timeout', str(args.wall_timeout)]
        argv += ['--worker-cpu', str(args.worker_cpu)]
        argv += ['--policy-path', str(artifacts.policy_path), '--deployment-path', str(artifacts.deployment_path),
                 '--training-gains-source', str(artifacts.training_source)]
        if artifacts.native_hold_source is not None:
            argv += ['--native-hold-source', str(artifacts.native_hold_source)]
        argv += ['--low-warmup-seconds', str(args.low_warmup_seconds)]
        argv += ['--projection-budget-rad', str(args.projection_budget_rad), '--raw-command-budget-rad', str(args.raw_command_budget_rad)]
        if args.project_targets:
            argv += ['--project-targets']
        if args.output_coordinates:
            argv += ['--output-coordinates']
        if args.saturate_output_velocity:
            argv += ['--saturate-output-velocity']
        if args.output_policy_joints_only:
            argv += ['--output-policy-joints-only']
        if args.project_encoder_targets:
            argv += ['--project-encoder-targets']
        if args.map_waist_gains:
            argv += ['--map-waist-gains']
        if args.serial_pd_torque:
            argv += ['--serial-pd-torque', '--serial-pd-delay-steps', *map(str, args.serial_pd_delay_steps)]
        if args.simulation_rate_limit is not None:
            argv += ['--simulation-rate-limit', str(args.simulation_rate_limit)]
        if args.backlash_limit_timeconst is not None:
            argv += ['--backlash-limit-timeconst', str(args.backlash_limit_timeconst)]
        script = ('export PYTHONPATH='+shlex.quote(str(ROOT/'env_isaaclab/lib/python3.12/site-packages'))
                  +':$PYTHONPATH\nexec '+shlex.join(argv))
        with (out/'worker.log').open('w') as log:
            result = run.shell(trial, script, stdout=log, stderr=subprocess.STDOUT, timeout=args.wall_timeout+60)
        report['worker_exit_code'] = result.returncode
        report['worker_result'] = json.loads((trial/'htd_result.json').read_text())
        report['pass_'] = bool(report['worker_result']['pass'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        if (trial/'htd_result.json').exists():
            report['worker_result'] = json.loads((trial/'htd_result.json').read_text())
        print(report['traceback'], flush=True)
    finally:
        if (trial/'runtime.json').exists():
            try:
                cfg = run.config(trial)
                stopped = subprocess.run(['docker', 'stop', '--timeout', '5', cfg['container']], capture_output=True, text=True, timeout=20)
                report['container_stop'] = dict(container=cfg['container'], exit_code=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr)
                if stopped.returncode:
                    report['pass_'] = False
            except Exception as exc:
                report['stop_error'] = str(exc)
                report['pass_'] = False
        report['pass'] = report.pop('pass_')
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HTD_NATIVE_PROBE', json.dumps({k: v for k, v in report.items() if k not in ('worker_result', 'source_sha256', 'traceback')}), flush=True)
    return 0 if report['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy-path', type=Path)
    parser.add_argument('--deployment-path', type=Path)
    parser.add_argument('--training-gains-source', type=Path)
    parser.add_argument('--native-hold-source', type=Path, help='Explicit hashed V2 carry hold source; includes six native wrist commands')
    parser.add_argument('--initial-state-file', type=Path, help='Fresh empty-hand one-reset native mechanism state; required for V2')
    parser.add_argument('--motion', type=Path, default=DEFAULT_MOTION)
    parser.add_argument('--image', default='igris-sim:jazzy')
    parser.add_argument('--physics-cpu', type=int, default=1)
    parser.add_argument('--worker-cpu', type=int, default=5)
    parser.add_argument('--map-waist-gains', action='store_true', help='Opt-in audited vendor zero-pose waist motor gains for the exact selected training profile; approximate serial PD only, defaults unchanged')
    parser.add_argument('--serial-pd-torque', action='store_true', help='Experimental opt-in: owned15 zero gains with independently scheduled 200Hz native serial PD effort from true output q/dq; no position-target clipping')
    parser.add_argument('--serial-pd-delay-steps', type=int, nargs=4, choices=[0, 1, 2], default=None,
                        metavar=('X12', 'X8', 'X6', 'PLA'), help='Required with serial PD: explicit fixed 5ms delays per vendor actuator group; nominal Isaac still samples 0..2')
    parser.add_argument('--simulation-rate-limit', type=float, default=None,
                        help='Opt-in serial PD wall pacing at finite rate (0,1], requested after reset before actor; native timestep and guards unchanged; default no pacing')
    parser.add_argument('--backlash-limit-timeconst', type=float, choices=[.005], default=None,
                        help='Explicit fresh-scene numerical variant: owned15 backlash solreflimit0.005 1; all ranges/solimp/mass/friction and guards unchanged; requires serial PD')
    parser.add_argument('--project-targets', action='store_true', help='Opt-in physical target projection within the explicit budget; preserves raw action history')
    parser.add_argument('--projection-budget-rad', type=float, default=.10, help='Aggregate physical-policy plus encoder projection budget; default0.10, explicit maximum0.30; encoder-only budget remains0.10')
    parser.add_argument('--raw-command-budget-rad', type=float, default=.10, help='Independent maximum raw-policy to published-command correction; default0.10, explicit maximum0.40; position-bias guard stays0.10')
    parser.add_argument('--output-coordinates', action='store_true', help='SIM-only native output q/dq and two-sided compensation using synchronized vendor FK; bounded correction and velocity bias')
    parser.add_argument('--saturate-output-velocity', action='store_true', help='Experimental opt-in: cap velocity compensation to +/-2rad/s, log raw/applied/saturation; loses exact output-PD identity while saturated')
    parser.add_argument('--output-policy-joints-only', action='store_true', help='Limit coordinate compensation to HTD15 waist/leg joints; all16 other targets and dq remain exactly raw SDK baseline; observer still logs31')
    parser.add_argument('--project-encoder-targets', action='store_true', help='Opt-in compensated PJS target projection; physical and encoder projection magnitudes share the explicit aggregate budget; encoder-only correction<=0.10; logs output residual')
    parser.add_argument('--low-warmup-seconds', type=float, default=0., help='Opt-in wall seconds of valid all31 default hold after LOW and before the single empty-hand SIM reset (vendor SlowStart nominally 3 s)')
    parser.add_argument('--stand-seconds', type=float, default=10.)
    parser.add_argument('--move-seconds', type=float, default=0.)
    parser.add_argument('--stop-seconds', type=float, default=5.)
    parser.add_argument('--command', type=float, nargs=3, default=[.03, 0., 0.])
    parser.add_argument('--wall-timeout', type=float, default=180.)
    args = parser.parse_args()
    if args.simulation_rate_limit is not None and (not math.isfinite(args.simulation_rate_limit) or not 0 < args.simulation_rate_limit <= 1):
        parser.error('Simulation wall-rate limit must be finite and within (0,1]')
    if args.simulation_rate_limit is not None and not args.serial_pd_torque:
        parser.error('This pacing option requires --serial-pd-torque')
    if args.backlash_limit_timeconst is not None and not args.serial_pd_torque:
        parser.error('The backlash numerical variant requires --serial-pd-torque')
    if args.serial_pd_torque:
        if not args.output_coordinates or args.serial_pd_delay_steps is None or args.low_warmup_seconds < 3.2:
            parser.error('Serial PD requires --output-coordinates, explicit --serial-pd-delay-steps, and LOW warmup >=3.2s')
        if any((args.project_targets, args.project_encoder_targets, args.map_waist_gains, args.saturate_output_velocity)):
            parser.error('Serial PD cannot combine with position projection, motor gain mapping or velocity compensation')
        if args.projection_budget_rad != .10 or args.raw_command_budget_rad != .10:
            parser.error('Position correction budgets do not apply to serial PD; leave defaults unchanged')
    elif args.serial_pd_delay_steps is not None:
        parser.error('Explicit PD delays require --serial-pd-torque')
    if args.saturate_output_velocity and not args.output_coordinates:
        parser.error('Velocity saturation requires --output-coordinates')
    if args.output_policy_joints_only and not args.output_coordinates:
        parser.error('Compensation ownership scope requires --output-coordinates')
    if args.project_encoder_targets and not args.output_policy_joints_only:
        parser.error('Encoder projection requires --output-policy-joints-only')
    if not np.isfinite([args.projection_budget_rad, args.raw_command_budget_rad]).all() or not 0 < args.projection_budget_rad <= .30 or not 0 < args.raw_command_budget_rad <= .40:
        parser.error('Projection budget must be within (0,0.30], raw-command budget within (0,0.40] rad')
    if args.projection_budget_rad != .10 and not args.project_targets:
        parser.error('An explicit projection budget requires --project-targets')
    if not np.isfinite([args.stand_seconds, args.move_seconds, args.stop_seconds, args.wall_timeout, args.low_warmup_seconds, *args.command]).all():
        parser.error('Finite duration and command values required')
    if not 0 < args.stand_seconds <= 60 or not 0 <= args.move_seconds <= 30 or not 0 <= args.stop_seconds <= 30 or args.wall_timeout <= 0 or not 0 <= args.low_warmup_seconds <= 10:
        parser.error('Invalid bounded probe durations')
    if np.any(np.abs(args.command) > np.array([.1, .07, .25])):
        parser.error('Command exceeds selected deployment limits')
    return (worker if args.action == 'worker' else launch)(args.out.resolve(), args)


if __name__ == '__main__':
    raise SystemExit(main())
