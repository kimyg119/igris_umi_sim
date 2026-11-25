"""SIM-only loaded arbitration; no ROS, services or physics.

The existing command loop must remain the sole LowCmd publisher. It supplies
fresh measured grasp/clearance evidence and live, already validated upper16
commands. This module requests LOW once, resets only actor/PD history, and
replaces exactly SDK0:15. It never requests HOME, torque-on, physics reset,
an initializer, or a SlowStart bypass. Default scopes remain ten-second zero
or goal-hold diagnostics. Explicit transport retains the same actor/PD clocks
through caller-authored lower/open targets, gated by measured support/release.

Use advance_actor(snapshot), reread native outputs/clocks after inference,
compute_pd(snapshot, upper_command), check_publication(frame, fresh_snapshot),
publish once, then ack_published(frame, fresh_snapshot). Do not resend a held
effort, run callbacks/file IO after computing PD, or create a second writer.
Catch ANY outer-loop error, publish abort(...).command once and immediately
pause native simulation while preserving hand force and live upper ownership.
Pausing and delivery acknowledgements remain the caller's responsibility.
"""
from dataclasses import dataclass
import copy
from numbers import Real

import numpy as np

from htd_policy import _immutable
from htd_serial_pd import vector


HELD8 = (15, 16, 17, 18, 22, 23, 24, 25)


def validate_upper_backlash_option(value, *, enabled, transport, current_grasp_reference,
                                  command_observer, wrist_mode_entry, goal_hold,
                                  contact_grace, relaxed_grasp=True):
    """A startup numerical candidate requires every explicit SIM prerequisite."""
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, Real)
            or value != .005 or not np.isfinite(value)):
        raise ValueError('Upper backlash time constant must be explicitly0.005')
    if not all(flag is True for flag in (enabled, transport, current_grasp_reference,
            command_observer, wrist_mode_entry, goal_hold, contact_grace, relaxed_grasp)):
        raise ValueError('Upper backlash requires explicit relaxed loaded transport/current-grasp/observer/wrist/retention scope')
    return float(value)


def upper_backlash_selection(manifest):
    """Require identical exact optional selection at manifest and loaded scope."""
    scope = manifest.get('htd_loaded_zero') or {}
    top_present = 'htd_upper_backlash_limit' in manifest
    scope_present = isinstance(scope, dict) and 'upper_backlash_limit' in scope
    if not top_present and not scope_present:
        return None
    from htd_backlash_limits import upper_descriptor
    expected = upper_descriptor(.005)
    def exact(value):
        return (type(value) is dict and value == expected
                and all(type(value[key]) is type(want) for key, want in expected.items())
                and all(type(index) is int for index in value['sdk_indices']))
    if (not top_present or not scope_present
            or not exact(manifest['htd_upper_backlash_limit'])
            or not exact(scope['upper_backlash_limit'])
            or manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend') != 'vendor MuJoCo 3.2.7'
            or manifest.get('evaluation_profile') != 'relaxed_transport'
            or scope.get('enabled') is not True or scope.get('scope') != 'loaded_transport'
            or scope.get('current_grasp_reference') is not True
            or any(not isinstance(scope.get(key), dict) for key in
                   ('goal_hold', 'command_observer', 'wrist_mode_entry', 'grasp_retention'))):
        raise ValueError('Upper backlash selection requires exact declared relaxed SIM transport scope')
    return .005


def validate_prealignment_coast_option(value, *, enabled, transport, current_grasp_reference,
                                       command_observer, wrist_mode_entry, goal_hold,
                                       contact_grace, relaxed_grasp=True):
    from htd_reviewed_approach import coast_descriptor
    if coast_descriptor(value) is None: return None
    if not all(flag is True for flag in (enabled, transport, current_grasp_reference,
            command_observer, wrist_mode_entry, goal_hold, contact_grace, relaxed_grasp)):
        raise ValueError('Prealignment coast requires explicit relaxed SIM transport/current-grasp/observer/wrist/retention scope')
    return 3.


def validate_prealignment_coast_policy(selection):
    from htd_reviewed_approach import COAST_ARTIFACTS,COAST_REPORT,COAST_REPORT_SHA256
    from pathlib import Path
    import hashlib
    if (not isinstance(selection,dict)
            or selection.get('selection')!='evaluated_posture_continuation'
            or selection.get('artifact_sha256')!=COAST_ARTIFACTS):
        raise ValueError('Coast requires the reviewed original5ecd58 four-artifact package')
    report=Path(__file__).resolve().parents[2]/COAST_REPORT
    if hashlib.sha256(report.read_bytes()).hexdigest()!=COAST_REPORT_SHA256:
        raise ValueError('Coast evidence report hash changed')


def prealignment_coast_selection(manifest):
    """Independent typed, source/evidence-pinned host/prestart/worker admission."""
    import json
    from pathlib import Path
    scope=manifest.get('htd_loaded_zero') or {}
    goal=scope.get('goal_hold') or {} if isinstance(scope,dict) else {}
    present=('htd_prealignment_coast' in manifest,
             isinstance(scope,dict) and 'prealignment_coast' in scope,
             isinstance(goal,dict) and 'prealignment_coast' in goal)
    if not any(present): return None
    from htd_reviewed_approach import coast_descriptor,COAST_REPORT,COAST_REPORT_SHA256
    expected=coast_descriptor(3.)
    def exact(value):
        try:
            return json.dumps(value,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False)
        except (ValueError,TypeError): return False
    if (not all(present) or not exact(manifest['htd_prealignment_coast'])
            or not exact(scope['prealignment_coast']) or not exact(goal['prealignment_coast'])
            or manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'
            or manifest.get('evaluation_profile')!='relaxed_transport'
            or scope.get('enabled') is not True or scope.get('scope')!='loaded_transport'
            or scope.get('current_grasp_reference') is not True
            or not isinstance(scope.get('transport'),dict)
            or scope['transport'].get('reviewed_approach') is not True
            or any(not isinstance(scope.get(k),dict) for k in ('command_observer','wrist_mode_entry','grasp_retention'))):
        raise ValueError('Coast selection requires exact declared reviewed relaxed SIM transport scope')
    validate_prealignment_coast_policy(scope.get('policy_selection'))
    report=str(Path(__file__).resolve().parents[2]/COAST_REPORT)
    if scope.get('setup_sha256',{}).get(report)!=COAST_REPORT_SHA256:
        raise ValueError('Coast evidence pin missing or changed in setup')
    return 3.


def validate_loaded_rate(rate, *, transport, current_grasp_reference):
    """Select wall pacing only; native actor/PD deadlines remain unchanged."""
    if isinstance(rate, bool) or not isinstance(rate, Real) or rate not in (.1, .05):
        raise ValueError('Loaded pacing rate must be finite numeric .1 or .05')
    if rate == .05 and not (transport is True and current_grasp_reference is True):
        raise ValueError('Loaded pacing .05 requires explicit transport/current-grasp scope')
    return float(rate)


def scoped_gc_configuration(params, robot):
    """Return fresh trial-local config: lower15 GC zero, upper16 unchanged.

The audited installed RobotManager LOW path calls computeFromStatus. This is
not the different bridge-command path that suppresses all GC at zero waist kp.
No live parameter mutation is authorized or performed by this helper.
"""
    if robot['igris_c']['controller']['GravityCompensationModule']['enabled'] is not True:
        raise ValueError('Existing upper16 GC must be enabled and preserved')
    result, robot_copy = copy.deepcopy(params), copy.deepcopy(robot)
    values = result['igris_c_param_server']['ros__parameters']
    for field in ('gravity_comp_gain', 'gravity_comp_offset'):
        source = vector(values[field], field).copy()
        source[:15] = 0.
        values[field] = source.tolist()
    return result, robot_copy


def validate_loaded_runtime(runtime, manifest, robot, params, reference_params):
    """Check supplied, already loaded trial configuration; no runtime mutation.

    Caller archives the exact YAML files and verifies robot.yaml's
    gain_parameter_path names this trial-local params file before launch.
    Native slowdown ACK is separately mandatory at request_handoff.
    """
    if (runtime.get('network') != 'none' or runtime.get('ipc') != 'private'
            or runtime.get('hardware_executed') is not False):
        raise ValueError('Private simulation runtime required')
    if ('htd_initial_state' in manifest or manifest.get('htd_loaded_zero', {}).get('enabled') is not True):
        raise ValueError('Explicit loaded-zero scope required; initializer is forbidden')
    if (runtime.get('simulation_rate_limit') != .5 or manifest.get('simulation_rate_limit') != .5
            or manifest.get('simulation_rate_start_phase_s') != 90.):
        raise ValueError('Preserve acquisition pacing0.5 from phase90; lower only after native ACK')
    if manifest.get('htd_backlash_limit_timeconst_s') != .005:
        raise ValueError('Require explicitly audited startup owned15 backlash variant0.005')
    upper_backlash_selection(manifest)
    expected, _ = scoped_gc_configuration(reference_params, robot)
    if params != expected:
        raise ValueError('Trial parameters must differ ONLY by lower15 GC gains/offsets zero')


@dataclass(frozen=True)
class LoadedSnapshot:
    native_time: float
    wall_time: float  # monotonic clock; every age below is measured at this time
    environment: int
    mode: int
    state_age_s: float
    mode_age_s: float
    joint_status: np.ndarray
    output: dict  # synchronized htd_output_coordinates sample, not SDK encoder q
    torso_quat_wxyz: np.ndarray  # corrected native IMU mount, NOT pelvis quaternion
    torso_gyro: np.ndarray
    pelvis_pose: np.ndarray  # world XYZ + WXYZ, used solely for fall guarding
    pose_age_s: float
    force_age_s: float
    hand_age_s: float
    hl_age_s: float
    grasp_verified: bool
    clearance_rise_m: float
    clearance_ready_s: float
    payload_speed_m_s: float
    source_support_z_N: float
    palm_tip_force_N: np.ndarray
    upper_output_target: np.ndarray  # current accepted HL physical target, before backlash compensation
    pacing_ack: dict | None  # native acknowledgement of the selected one-shot slowdown from0.5
    torso_pose: np.ndarray | None = None  # world XYZ/WXYZ for opt-in feedback
    pose_native_time: float | None = None
    force_native_time: float | None = None
    payload_pose: np.ndarray | None = None
    hand_poses: np.ndarray | None = None
    transport_phase: str = 'carry'
    destination_support_z_N: float = 0.
    placement_target_xyz: np.ndarray | None = None
    placement_done: bool = False
    phase_target_valid: bool = False


@dataclass(frozen=True)
class LoadedFrame:
    command: dict
    pd: object
    measured_waist: np.ndarray
    epoch: int


@dataclass(frozen=True)
class LoadedCleanup:
    command: dict
    reason: str
    pause_required: bool = True
    hold_grip: bool = True


def _commands(command):
    result = {key: vector(command[key], key).copy() for key in ('q', 'dq', 'kp', 'kd', 'tau')}
    if np.any(result['kp'] < 0) or np.any(result['kd'] < 0):
        raise ValueError('Negative command gains')
    if 'kinematic_modes' in command:
        modes=np.asarray(command['kinematic_modes'])
        if modes.shape!=(5,) or not np.isin(modes,[0,1]).all() or not np.all(modes[:3]==1):
            raise ValueError('Loaded kinematic modes must preserve owned15 PJS groups')
        result['kinematic_modes']=modes.astype(int,copy=True)
    return result


def _frozen(command):
    return {key: _immutable(value) for key, value in command.items()}


class LoadedZero:
    """Bounded coordinator. Readiness is evidence, not a loaded-stability claim.

    Safety bounds retain native24's pelvis0.7m/tilt0.4rad, serial caps and
    output-bias0.1rad. Added loaded guards require both friction-qualified
    palms continuously, fresh hand/HL evidence, physical upper14 error<=0.2rad,
    and accepted held8 within0.1rad of the frozen proxy. An explicit SIM-only
    current-grasp diagnostic captures that reference once at qualified handoff
    and records its difference from the frozen proxy without changing weights.
    These two upper bounds
    are diagnostic guards, not an assertion of exact proxy posture equivalence.
    """
    def __init__(self, policy, serial_pd, hold_q, *, mass_kg, friction,
                 clearance_lift_m, duration_s=10., current_grasp_reference=False,
                 clearance_minimum_rise_m=None, goal_hold=False, contact_grace=False,
                 transport_target_xyz=None, transport_heading=None, reviewed_approach=False, loaded_rate=.1, prealignment_coast_seconds=None):
        self.transport = transport_target_xyz is not None or transport_heading is not None
        if prealignment_coast_seconds is not None:
            validate_prealignment_coast_option(prealignment_coast_seconds,enabled=True,
                transport=self.transport,current_grasp_reference=current_grasp_reference,
                command_observer=True,wrist_mode_entry=True,goal_hold=goal_hold,contact_grace=contact_grace)
            from htd_reviewed_approach import COAST_ARTIFACTS
            if reviewed_approach is not True or policy.policy_sha256!=COAST_ARTIFACTS['policy.ts']:
                raise ValueError('Coast coordinator requires reviewed approach and original policy')
        self.loaded_rate = validate_loaded_rate(loaded_rate, transport=self.transport,
            current_grasp_reference=current_grasp_reference)
        if type(reviewed_approach) is not bool:
            raise ValueError('Reviewed approach requires an explicit boolean')
        if reviewed_approach and not self.transport:
            raise ValueError('Reviewed approach requires explicit transport scope')
        self.reviewed_approach = reviewed_approach
        if self.transport and (transport_target_xyz is None or transport_heading is None
                or goal_hold is not True or current_grasp_reference is not True
                or duration_s not in (10., 180.)):
            raise ValueError('Explicit transport target/heading/current-grasp goal-hold scope required')
        if self.transport: duration_s = 180.
        if (not np.isfinite([mass_kg, friction, clearance_lift_m, duration_s]).all()
                or mass_kg <= 0 or friction <= 0 or not 0 < clearance_lift_m <= .06
                or not 0 < duration_s <= (180. if self.transport else 10.)):
            raise ValueError('Bounded loaded-zero physical configuration required')
        if type(current_grasp_reference) is not bool:
            raise ValueError('Current-grasp scope requires an explicit boolean')
        if type(goal_hold) is not bool or (goal_hold and not current_grasp_reference):
            raise ValueError('Goal hold requires explicit current-grasp scope')
        if type(contact_grace) is not bool or (contact_grace and not goal_hold):
            raise ValueError('Contact grace requires explicit goal hold scope')
        self.grasp_retention=None
        if contact_grace:
            from htd_grasp_retention import GraspRetention
            self.grasp_retention=GraspRetention(mass_kg,friction,
                qualified_translation_limit_m=.025 if self.transport else .015)
        self.goal_hold = None
        if goal_hold:
            from htd_goal_hold import GoalHold
            self.goal_hold = GoalHold()
        self.placement = None
        if self.transport:
            from htd_navigation_feedback import NavigationFeedback
            from htd_transport_phase import TransportPlacement
            self.goal_hold = NavigationFeedback(transport_target_xyz, transport_heading,
                                                reviewed_approach=reviewed_approach,
                                                prealignment_coast_seconds=prealignment_coast_seconds)
            self.placement = TransportPlacement(transport_target_xyz, mass_kg,friction=friction,
                                               qualified_translation_limit_m=.025)
        if clearance_minimum_rise_m is not None:
            minimum_rise = .015 if goal_hold else .02
            if (not current_grasp_reference or not np.isfinite(clearance_minimum_rise_m)
                    or not minimum_rise <= clearance_minimum_rise_m <= .8*clearance_lift_m):
                raise ValueError('Explicit current-grasp scope required for bounded clearance override')
        self.current_grasp_reference = current_grasp_reference
        self.clearance_minimum_rise_m = (.8*clearance_lift_m if clearance_minimum_rise_m is None
                                         else float(clearance_minimum_rise_m))
        if policy.dt != .02 or serial_pd.dt != .005:
            raise ValueError('Require 50Hz actor and 200Hz serial PD')
        if serial_pd.contract['configured_delay_steps'] != [0, 0, 0, 0]:
            raise ValueError('Loaded diagnostic requires explicit serial delay0000')
        self.policy, self.serial_pd = policy, serial_pd
        self.hold_q = _immutable(vector(hold_q, 'hold_q'))
        if not np.array_equal(self.hold_q, serial_pd.contract['feasible_command_q']):
            raise ValueError('Serial PD hold differs from frozen profile')
        self._held_reference = None if current_grasp_reference else _immutable(self.hold_q[list(HELD8)])
        self.current_grasp_evidence = None
        self.required_palm_force = 1.5*mass_kg*9.81/(2*friction)
        self.mass_kg, self.clearance_lift_m, self.duration_s = mass_kg, clearance_lift_m, duration_s
        self.state = 'prepared'
        self.epoch = 0
        self.origin = self.requested_wall = self.requested_native = None
        self.raw_target = self._pending = None
        self._last_published_command = None
        self.actor_count = self.published_count = 0
        self.events = []
        self.contract = dict(scope='SIM-only loaded zero velocity for at most10 native seconds; no transport claim',
            actor_dt_s=.02, pd_dt_s=.005, duration_s=duration_s, commanded_velocity=[0., 0., 0.],
            ownership='existing sole LowCmd writer; replace SDK0:15 only; preserve live upper16 fields',
            gc='enabled; gains and offsets0:15 zero; preserve original upper16',
            slow_start='LOW entry reactivates upper14 q slew; no bypass or inferred expiration',
            upper_physical_error_guard_rad=.2, held8_proxy_target_guard_rad=.1,
            current_grasp_reference=current_grasp_reference, runtime_domain_validated=False,
            held8_reference='capture accepted current grasp once at qualified handoff' if current_grasp_reference else 'frozen profile76',
            clearance_minimum_rise_m=self.clearance_minimum_rise_m,
            domain_assessment='Unvalidated acquired grasp; policy weights identified by runtime selection, owned15 offsets/caps unchanged' if current_grasp_reference else 'Frozen profile76 runtime correction; not a loaded-domain validation',
            resets='actor history/phase and explicit PD buffer only; never physics or initializer',
            completion='zero owned effort then pause; hold grip and upper control; no automatic WALK_LOW fallback')
        self.contract['grasp_contact_rule'] = ('per-hand palm + tip' if goal_hold else 'per-hand palm only')
        if goal_hold:
            self.contract.update(scope='SIM-only loaded XY/heading goal hold for at most10 native seconds; no transport claim',
                commanded_velocity='WorldXY goal/heading feedback', goal_hold=self.goal_hold.contract)
        if contact_grace:
            self.contract.update(grasp_contact_rule='per-hand palm+tip, with explicit force/pose guarded100ms grace',
                                 grasp_retention=self.grasp_retention.contract)
        if self.transport:
            self.contract.update(scope='SIM explicit loaded navigation and supported placement; timeout is failure',
                transport=self.placement.contract, goal_hold=self.goal_hold.contract,
                commanded_velocity='Continuous NavigationFeedback through carry/lower/open/done',
                grasp_contact_rule='Existing carry/unsupported-lower friction rule; supported lower retains relative geometry; open/done require measured receiver support and bounded payload',
                completion='Measured supported opening + explicit placement_done + full PD coverage, then pause; preserve last upper state')
        if reviewed_approach:
            self.contract['reviewed_approach'] = True

    def _check(self, s, *, require_fresh_output=False):
        if s.environment != 1:
            raise RuntimeError('SIM RobotState required; hardware is never supported')
        for name, limit in [('state_age_s', .1), ('mode_age_s', .5), ('pose_age_s', .5),
                            ('force_age_s', .25), ('hand_age_s', .25), ('hl_age_s', .25)]:
            value = getattr(s, name)
            if not np.isfinite(value) or not 0 <= value < limit:
                raise RuntimeError('Stale/invalid '+name)
        if np.any(vector(s.joint_status, 'joint_status')):
            raise RuntimeError('SDK joint status error')
        if not np.isfinite([s.native_time, s.wall_time]).all() or s.native_time < 0:
            raise RuntimeError('Invalid native/monotonic time')
        output = s.output
        q = vector(output['q_output'], 'q_output')
        for key in ('dq_output', 'q_encoder', 'dq_encoder'):
            vector(output[key], key)
        if np.max(np.abs(q[:15]-np.asarray(output['q_encoder'])[:15])) > .10+1e-12:
            raise ValueError('Owned output position bias exceeds0.10rad')
        actual = self.hold_q.copy()
        actual[:15] = q[:15]
        # Reuse exactly the adapter's physical URDF/coupled-polytope gate BEFORE
        # requesting LOW as well as during active PD; no target projection.
        self.serial_pd._gate.check(actual)
        quat, gyro = np.asarray(s.torso_quat_wxyz), np.asarray(s.torso_gyro)
        if (quat.shape != (4,) or gyro.shape != (3,) or not np.isfinite(quat).all()
                or not np.isfinite(gyro).all() or abs(np.linalg.norm(quat)-1) > 1e-3):
            raise ValueError('Finite mounted torso IMU with unit WXYZ quaternion required')
        if (not np.isfinite([output['sim'], output['monotonic']]).all()
                or not -1e-9 <= s.native_time-output['sim'] < (.002 if require_fresh_output else .01)
                or not 0 <= s.wall_time-output['monotonic'] < .1
                or not isinstance(output['sequence'], (int, np.integer))
                or output['sequence'] <= 0 or output['sequence'] % 2):
            raise RuntimeError('Fresh complete native output sample required')
        pose = np.asarray(s.pelvis_pose)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-3:
            raise RuntimeError('Invalid pelvis pose')
        tilt = np.arccos(np.clip(1-2*(pose[4]**2+pose[5]**2), -1, 1))
        if pose[2] < .7 or tilt > .4:
            raise RuntimeError('Native pelvis height/tilt fall guard')
        forces = np.asarray(s.palm_tip_force_N)
        if (forces.shape != (4,) or not np.isfinite(forces).all() or s.grasp_verified is not True
                or np.any(forces < 0)):
            raise RuntimeError('Measured friction-qualified grasp required')
        qualified = forces[[0, 2]] if self.goal_hold is None else forces.reshape(2, 2).sum(axis=1)
        contact_required = True
        if self.transport:
            if s.transport_phase != 'carry' and (self.state != 'active'
                    or self._pending is not None and s.transport_phase != self.placement.phase):
                raise RuntimeError('Transport phase transition requires active ownership and no unpublished PD frame')
            self.placement.check(s, navigation_ready=self.goal_hold.success)
            contact_required = self.placement.hand_contact_required
        elif s.transport_phase != 'carry' or s.placement_done:
            raise RuntimeError('Explicit transport scope required for placement phases')
        if contact_required and self.grasp_retention is not None and self.grasp_retention.captured:
            self.grasp_retention.check(s)
        elif contact_required and np.any(qualified < self.required_palm_force):
            raise RuntimeError('Measured friction-qualified grasp required')
        upper = vector(s.upper_output_target, 'upper_output_target')
        if (not self.transport or s.transport_phase == 'carry') and self._held_reference is not None and np.max(np.abs(upper[list(HELD8)]-self._held_reference)) > .1:
            reference = 'captured current grasp' if self.current_grasp_reference else 'frozen profile'
            raise RuntimeError('Accepted upper held8 departed '+reference)
        if np.max(np.abs(q[15:29]-upper[15:29])) > .2:
            raise RuntimeError('Physical upper14 target error exceeds0.2rad during SlowStart/hold')
        if self.state == 'active' and s.mode != 3:
            raise RuntimeError('LOW ownership lost')

    def check_handoff_ready(self, s, *, require_fresh_output=True):
        """Validate a prospective handoff without changing ownership or actor state."""
        if self.state != 'prepared':
            raise RuntimeError('Handoff may only be requested once from prepared')
        self._check(s, require_fresh_output=require_fresh_output)
        if s.mode != 6:
            raise RuntimeError('Measured WALK_LOW acquisition required')
        ack = s.pacing_ack
        if (not isinstance(ack, dict) or any(k not in ack for k in
                ('rate', 'previous_rate', 'native_time_s', 'monotonic_s'))
                or not np.isfinite(list(ack[k] for k in ('rate', 'previous_rate', 'native_time_s', 'monotonic_s'))).all()
                or ack['rate'] != self.loaded_rate or ack['previous_rate'] != .5
                or not 0 <= ack['native_time_s'] <= s.native_time
                or not 0 <= ack['monotonic_s'] <= s.wall_time):
            raise RuntimeError(f'Native pacing slowdown acknowledgement0.5 ->{self.loaded_rate:g} required')
        clearance = [s.clearance_rise_m, s.clearance_ready_s, s.payload_speed_m_s, s.source_support_z_N]
        if (not np.isfinite(clearance).all() or s.clearance_rise_m < self.clearance_minimum_rise_m
                or s.clearance_ready_s < .5 or not 0 <= s.payload_speed_m_s < .02
                or s.source_support_z_N >= .2*self.mass_kg*9.81):
            raise RuntimeError('Measured stable source clearance required')

    def request_handoff(self, s):
        self.check_handoff_ready(s)
        if self.placement is not None:
            self.placement.capture(s)
        if self.grasp_retention is not None:
            self.grasp_retention.capture(s)
        if self.goal_hold is not None:
            self.goal_hold.capture(s)
        if self.current_grasp_reference:
            self._held_reference = _immutable(np.asarray(s.upper_output_target)[list(HELD8)])
            delta = self._held_reference-self.hold_q[list(HELD8)]
            self.current_grasp_evidence = dict(native_time_s=s.native_time, monotonic_s=s.wall_time,
                sdk_indices=list(HELD8), accepted_target_rad=self._held_reference.tolist(),
                measured_output_rad=np.asarray(s.output['q_output'])[list(HELD8)].tolist(),
                frozen_profile_target_rad=self.hold_q[list(HELD8)].tolist(),
                delta_from_frozen_profile_rad=delta.tolist(),
                max_abs_delta_from_frozen_profile_rad=float(np.max(np.abs(delta))),
                runtime_domain_validated=False, clearance_rise_m=s.clearance_rise_m,
                clearance_minimum_rise_m=self.clearance_minimum_rise_m,
                reference_capture='Once, after all measured handoff checks; never rebased')
            self.events.append(dict(event='captured_current_grasp_reference', **self.current_grasp_evidence))
        self.requested_wall, self.requested_native = s.wall_time, s.native_time
        self.state = 'low_pending'
        self.epoch += 1
        self.events.append(dict(event='request_low', native=s.native_time, monotonic=s.wall_time, epoch=self.epoch))
        return dict(mode_command_type=5, preset_id='', is_cyclic=False)

    def advance_actor(self, s):
        if self.state == 'complete':
            return None
        if self.state not in ('low_pending', 'active'):
            raise RuntimeError('Actor cannot run while '+self.state)
        self._check(s)
        if self.state == 'low_pending':
            if (not 0 <= s.wall_time-self.requested_wall < .5
                    or not -1e-9 <= s.native_time-self.requested_native < .05):
                raise RuntimeError('LOW transition acknowledgement timeout; do not wait for SlowStart')
            if s.mode == 6:
                return None
            if s.mode != 3:
                raise RuntimeError('Unexpected mode during LOW transition')
            if s.native_time-s.output['sim'] >= .002:
                return None  # fresh300Hz sample required before establishing the clock origin
            self.origin = float(s.output['sim'])
            self.policy.reset(sim_time=self.origin)
            self.serial_pd.reset(self.origin)
            self.state = 'active'
            self.events.append(dict(event='low_confirmed_actor_history_reset', native=self.origin, epoch=self.epoch))
        if s.native_time-self.origin >= self.duration_s-1e-9:
            if self.transport:
                raise RuntimeError('Loaded transport timeout; placement incomplete')
            if self._pending is not None or self.published_count != int(np.ceil(self.duration_s/.005-1e-9)):
                raise RuntimeError('Incomplete PD publication coverage at stop')
            self.state = 'complete'
            self.events.append(dict(event=('bounded_goal_hold_complete_pause_required' if self.goal_hold is not None
                                           else 'bounded_zero_complete_pause_required'), native=s.native_time))
            return None
        if self.transport and self.placement.completion_ready and self._pending is None:
            expected = int(np.ceil((s.native_time-self.origin)/.005-1e-9))
            if self.published_count == expected:
                self.state = 'complete'
                self.events.append(dict(event='loaded_transport_complete_pause_required', native=s.native_time,
                    actor_count=self.actor_count, published_count=self.published_count,
                    placement_metrics=copy.deepcopy(self.placement.metrics)))
                return None
        if not self.serial_pd.due(s.output, native_time=s.native_time, wall_time=s.wall_time):
            return None
        if not self.policy.policy_due(s.output['sim']):
            return None
        command = np.zeros(3) if self.goal_hold is None else self.goal_hold.step(s, self.policy.dt)
        result = self.policy.step(s.output['q_output'], s.output['dq_output'],
            s.torso_quat_wxyz, s.torso_gyro, command, s.output['sim'])
        self.raw_target = self.hold_q.copy()
        self.raw_target[:15] = result.q_target[:15]
        self.actor_count += 1
        return result

    def compute_pd(self, s, upper_command):
        if self.state in ('low_pending', 'complete'):
            return None
        if self.state != 'active':
            raise RuntimeError('PD cannot run while '+self.state)
        self._check(s)
        if self._pending is not None:
            raise RuntimeError('Previous PD frame remains unpublished')
        if not self.serial_pd.due(s.output, native_time=s.native_time, wall_time=s.wall_time):
            return None
        if self.raw_target is None:
            raise RuntimeError('Fresh actor reference required before PD')
        if self.policy.policy_due(s.output['sim']):
            raise RuntimeError('Due actor must run before computing the next PD frame')
        command = _commands(upper_command)
        record = self.serial_pd.step(self.raw_target, s.output, native_time=s.native_time, wall_time=s.wall_time)
        for key in ('q','dq','kp','kd','tau'):
            command[key][:15] = getattr(record, key)[:15]
        self._pending = LoadedFrame(_frozen(command), record, _immutable(np.asarray(s.output['q_output'])[:3]), self.epoch)
        return self._pending

    def check_publication(self, frame, s):
        if frame is not self._pending or self.state != 'active' or frame.epoch != self.epoch:
            raise RuntimeError('PD frame already published or ownership epoch changed')
        self._check(s)
        self.serial_pd.check_publication(frame.pd, native_time=s.native_time, wall_time=s.wall_time)

    def remember_published(self, frame):
        """Record a successful publisher call before any fallible clock read."""
        if frame is not self._pending or self.state != 'active' or frame.epoch != self.epoch:
            raise RuntimeError('PD frame already published or ownership epoch changed')
        self._last_published_command = frame.command

    def ack_published(self, frame, s):
        # Repeating remember here retains compatibility with callers that have
        # an infallible postpublication clock; runtime calls it before clock IO.
        self.remember_published(frame)
        self.check_publication(frame, s)
        self._pending = None
        self.published_count += 1

    def abort(self, reason, upper_command):
        """Publish once then pause, preserving the last upper/grip or supported-open state."""
        command = _commands(self._last_published_command if self._last_published_command is not None else upper_command)
        command['q'][:15] = self.hold_q[:15]
        for key in ('dq', 'kp', 'kd', 'tau'):
            command[key][:15] = 0.
        self.state = 'paused'
        self._pending = None
        released = self.transport and self.placement.phase in ('open', 'done')
        self.events.append(dict(event='pause_required_preserve_open' if released else 'pause_required_hold_grip',
                                reason=str(reason), epoch=self.epoch))
        return LoadedCleanup(_frozen(command), str(reason), hold_grip=not released)
