"""Explicit SIM-only route reviewed against recorded native105 geometry.

These helpers emit goals/commands and diagnostic failures, never physical state
changes. Static clearance of recorded poses does not validate new gait dynamics.
The default V5/HTD route and the shared WBC controller remain unchanged.
"""
from dataclasses import replace
import math
import copy
from numbers import Real

import torch

import numpy as np

from heading_route import HeadingRoute
from htd_goal_hold import GoalHold, _wrap, _yaw, world_xy
from htd_navigation import NavigationGoals


GEOMETRY_REVIEW_SCHEMA = 'native105_static_approach_envelope/v1'
GEOMETRY_REVIEW_REPORT = 'outputs/hlik_transport_20260918_2005/approach105_envelope01/REPORT.md'
GEOMETRY_REVIEW_REPORT_SHA256 = 'fbb83729aea1b8c0d78cf0fa5ec366b5388230be210dfd779554d42900517a57'
GEOMETRY_REVIEW_DATA = 'outputs/hlik_transport_20260918_2005/approach105_envelope01/envelope.json'
GEOMETRY_REVIEW_DATA_SHA256 = 'ac671204b147ac0417299ac9a9f9de3f9fb2d33b0943d47e6cce3c8b18bc7583'


def geometry_review():
    """Immutable artifact pins copied into evidence; no runtime filesystem IO."""
    return dict(schema=GEOMETRY_REVIEW_SCHEMA, native_trial=105,
                report=GEOMETRY_REVIEW_REPORT, report_sha256=GEOMETRY_REVIEW_REPORT_SHA256,
                data=GEOMETRY_REVIEW_DATA, data_sha256=GEOMETRY_REVIEW_DATA_SHA256,
                dynamic_validation='pending',
                fresh_capture_geometry_validated=False,
                runtime_capture_anchor_gate=False,
                fresh_capture_limitation='No runtime gate ties fresh capture start, heading or goal to '
                    'the105 world anchors; static105 clearance does not guarantee a fresh capture path',
                prealignment_limitation='The X band qualifies dwell only; leaving it does not '
                    'immediately abort lateral alignment. The30-native-second timeout bounds this '
                    'SIM diagnostic, including potentially unreviewed prealignment excursions',
                limitation='Original105 scene and recorded pre-contact posture envelope only; '
                           'new policy dynamics, acquisition, lowering and release are unverified')


COAST_REPORT = 'outputs/hlik_transport_20260918_2005/native111_feedback_followup_review01/REPORT.md'
COAST_REPORT_SHA256 = '9faef74f70995976f13da89234933653802a9beb5ace9217da9c53f591a29f91'
COAST_ARTIFACTS = {
    'model.pt': '5ecd58255fa2c86190f5a2dce92652e999900a5183f9a28a33eb4f7a98be6ee0',
    'policy.ts': 'c0f6181e5721752314774406b24b4580a07e35715d8b4eb2a21d6df4e0210a6a',
    'deployment.json': '0e1e76b6add6b84792c2e1716705e264c53d6c42ab1ba9ba186d800d52c56802',
    'runtime_env.yaml': 'f273dc2509f65e2d2ea41fd3bdee48c064cd0881f97ae7ac0673e03317505859',
}


def coast_descriptor(seconds):
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or seconds != 3.:
        raise ValueError('Prealignment coast must be explicit numeric3.0 seconds')
    return dict(schema='htd_prealignment_coast/v1', simulation_only=True,
                duration_native_s=3., once=True, report=COAST_REPORT,
                report_sha256=COAST_REPORT_SHA256, artifact_sha256=dict(COAST_ARTIFACTS),
                admission='Original instantaneous root XY/heading/component-velocity gates; advancing fresh pose; prior emitted standing command; remaining timeout',
                timer='Starts at emitted exact XYZ zero after unchanged vector/angular slew',
                route='Original measured dwell/timeouts remain active; leaving prealignment censors coast',
                history='Base update once per due actor; no filter/arrival/settle/actor/PD reset',
                validation='SIM diagnostic only; no improvement or clearance claim')


class ReviewedApproachError(RuntimeError):
    """Latched diagnostic routed to the caller's existing stop/pause handling."""


class ReviewedApproachRoute(HeadingRoute):
    ALIGNMENT_TIMEOUT_S = 30.
    ALIGNMENT_DWELL_S = .3
    ALIGNMENT_X = (.63, .67)
    LATERAL_TOLERANCE_M = .015
    OVERSHOOT_LIMIT_M = .030

    def __init__(self, start_xy, final_xy, heading, *, native_time, pose_native_time):
        super().__init__(start_xy, final_xy, heading)
        self.heading = float(heading)
        self.final_local = self.rotation.T @ (np.asarray(final_xy)-self.start)
        if self.final_local[0] <= self.ALIGNMENT_X[1]:
            raise ValueError('Final goal must be beyond the reviewed alignment corridor')
        self.waypoints = [self.start+self.rotation@p for p in
                          ([0., .085], [.65, .085], [.65, self.final_local[1]])]
        self.waypoints.append(np.asarray(final_xy, dtype=float).copy())
        self._last_native = self._pose_time = None
        self._position = self.start.copy()
        self._last_heading = self.heading
        self._validate_clock(native_time, pose_native_time)
        self._last_native, self._pose_time = float(native_time), float(pose_native_time)
        self.root_velocity_route = np.zeros(2)
        self.root_velocity_world = np.zeros(2)
        self.alignment_started_native = None
        self._qualified_since_pose = None
        self.qualified_duration_s = 0.
        self.failure = None
        self.pose_advanced = False

    @property
    def final(self):
        return self.index == 3

    def _validate_clock(self, native, stamp):
        if not np.isfinite([native, stamp]).all() or min(native, stamp) < 0:
            raise ValueError('Finite nonnegative reviewed-route clocks required')
        if not -1e-9 <= native-stamp <= .05+1e-9:
            raise ValueError('Reviewed-route pose is future or older than .05 seconds')
        if self._last_native is not None and (native < self._last_native or stamp < self._pose_time):
            raise ValueError('Reviewed-route native or pose clock moved backward')

    def _fail(self, code, native, local):
        self.failure = dict(code=code, native_time=float(native),
                            pose_native_time=float(self._pose_time),
                            measured_local_xy=local.tolist(),
                            final_local_xy=self.final_local.tolist(),
                            waypoint_index=self.index)
        raise ReviewedApproachError(f'{code}: measured root local XY={local.tolist()}')

    def advance(self, position, *, heading, native_time, pose_native_time):
        if self.failure is not None:
            raise ReviewedApproachError(self.failure['code']+': reviewed approach failure is latched')
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite([*position, heading]).all():
            raise ValueError('Finite measured root XY and heading required')
        self._validate_clock(native_time, pose_native_time)
        if self._pose_time == pose_native_time and (
                not np.array_equal(position, self._position)
                or abs(_wrap(heading-self._last_heading)) > 1e-12):
            raise ValueError('Changed reviewed-route pose at identical pose stamp')
        pose_dt = float(pose_native_time)-self._pose_time
        self.pose_advanced = 0 < pose_dt <= .05+1e-9
        if pose_dt > 0:
            self.root_velocity_world = (position-self._position)/pose_dt
            self.root_velocity_route = self.rotation.T@self.root_velocity_world
        self._position, self._last_heading = position.copy(), float(heading)
        self._last_native, self._pose_time = float(native_time), float(pose_native_time)
        local = self.rotation.T@(position-self.start)
        eps = 1e-12  # Arithmetic roundoff only; the geometric tolerances are unchanged.
        if self.final:
            if local[0] < self.ALIGNMENT_X[0]-eps:
                self._fail('final_source_backtrack', native_time, local)
            if local[0] > self.final_local[0]+self.OVERSHOOT_LIMIT_M+eps:
                self._fail('final_forward_overshoot', native_time, local)
            if abs(local[1]-self.final_local[1]) > self.LATERAL_TOLERANCE_M+eps:
                self._fail('final_lateral_corridor_exit', native_time, local)
            return False
        if self.index == 2:
            if native_time-self.alignment_started_native > self.ALIGNMENT_TIMEOUT_S+eps:
                self._fail('alignment_timeout', native_time, local)
            qualified = all(self.qualification_flags().values())
            # Sparse calls must not turn an unobserved interval into a dwell.
            # Runtime normally supplies a fresh measurement every20ms.
            if pose_dt > .05+1e-9:
                self._qualified_since_pose = None
                self.qualified_duration_s = 0.
            if qualified:
                if self._qualified_since_pose is None:
                    self._qualified_since_pose = float(pose_native_time)
                self.qualified_duration_s = float(pose_native_time)-self._qualified_since_pose
                if self.qualified_duration_s >= self.ALIGNMENT_DWELL_S-eps:
                    self.index = 3
                    return True
            else:
                self._qualified_since_pose = None
                self.qualified_duration_s = 0.
            return False
        if not self.minimum_departure-eps <= local[1] <= .20+eps:
            return False
        if self.index == 0:
            if abs(local[0]) > .045+eps:
                return False
            self.waypoints[1] = self.start+self.rotation@np.array([.65, max(.085, local[1])])
        elif not self.ALIGNMENT_X[0]-eps <= local[0] <= self.ALIGNMENT_X[1]+eps:
            return False
        self.index += 1
        if self.index == 2:
            self.alignment_started_native = float(native_time)
        return True

    def qualification_flags(self):
        local = self.rotation.T@(self._position-self.start)
        eps = 1e-12
        return dict(x=self.ALIGNMENT_X[0]-eps <= local[0] <= self.ALIGNMENT_X[1]+eps,
                    lateral=abs(local[1]-self.final_local[1]) <= self.LATERAL_TOLERANCE_M+eps,
                    heading=abs(_wrap(self._last_heading-self.heading)) <= .08+eps,
                    velocity=np.max(np.abs(self.root_velocity_route)) <= .03+eps)

    def evidence(self):
        return dict(schema='htd_reviewed_approach_route/v1', waypoint_index=self.index,
                    phase=('source_departure', 'source_forward', 'prealignment', 'final')[self.index],
                    final_waypoint=self.final, start_xy=self.start.tolist(),
                    waypoints_xy=[p.tolist() for p in self.waypoints],
                    minimum_measured_departure_m=self.minimum_departure,
                    source_forward_gate_m=list(self.ALIGNMENT_X),
                    prealignment_local_x_m=.65, final_local_xy=self.final_local.tolist(),
                    intermediate_settling_required=True,
                    alignment_timeout_s=self.ALIGNMENT_TIMEOUT_S,
                    alignment_started_native=self.alignment_started_native,
                    qualified_duration_s=self.qualified_duration_s,
                    qualification=dict(x_range_m=list(self.ALIGNMENT_X),
                                       lateral_error_m=self.LATERAL_TOLERANCE_M,
                                       heading_error_rad=.08, each_root_velocity_component_m_s=.03,
                                       velocity_frame='captured pelvis heading',
                                       duration_native_pose_s=self.ALIGNMENT_DWELL_S,
                                       maximum_dwell_pose_gap_s=.05,
                                       repeated_pose_stamp='retain velocity; no additional dwell'),
                    root_velocity_world_m_s=self.root_velocity_world.tolist(),
                    root_velocity_route_m_s=self.root_velocity_route.tolist(),
                    final_corridor=dict(minimum_local_x_m=self.ALIGNMENT_X[0],
                                        maximum_local_x_m=float(self.final_local[0]+self.OVERSHOOT_LIMIT_M),
                                        lateral_error_m=self.LATERAL_TOLERANCE_M),
                    failure=None if self.failure is None else dict(self.failure),
                    geometry_review=geometry_review())


class ReviewedApproachController(world_xy.WorldXYController):
    """Asymmetric desired final speed; base acceleration/filter/history retained.

The commanded speed can exceed .04 temporarily while decelerating from a
previous phase. This is a desired-speed cap, never an instantaneous clipping.
"""
    def __init__(self, count, device='cpu', cfg=None, *, prealignment_coast_seconds=None):
        if count != 1:
            raise ValueError('Reviewed approach requires a single controller')
        super().__init__(count, device=device, cfg=cfg)
        self.reviewed_final_phase = False
        self.prealignment_coast_seconds = (None if coast_descriptor(prealignment_coast_seconds) is None else 3.)
        self.route_context = None
        self._coast_last_native = None
        self._coast = dict(state='armed', trigger_native=None, zero_started_native=None,
                           ended_native=None, zero_elapsed_s=0., reason=None)

    def coast_evidence(self):
        return copy.deepcopy(self._coast)

    def censor_coast(self, reason, native_time=None):
        if self.prealignment_coast_seconds is None or self._coast['state'] in ('completed','censored','not_triggered'):
            return
        self._coast['state'] = 'not_triggered' if self._coast['state']=='armed' else 'censored'
        self._coast.update(reason=str(reason), ended_native=native_time,
                           last_observed_native=self._coast_last_native)
        if native_time is not None and self._coast['zero_started_native'] is not None:
            self._coast['zero_elapsed_s'] = max(0., native_time-self._coast['zero_started_native'])

    def _coast_update(self, previous, nominal, desired, dt):
        r = self.route_context
        now = r._last_native
        e = self._coast
        flags = {k:bool(v) for k,v in r.qualification_flags().items()}
        standing = bool((previous[0,:2].norm() <= .005) & (previous[0,2].abs() <= .01))
        remaining = None if r.alignment_started_native is None else r.ALIGNMENT_TIMEOUT_S-(now-r.alignment_started_native)
        acceleration = self.cfg.braking_accel if bool(self.arrived[0]) else self.cfg.linear_accel
        ramp_ticks = math.ceil(max(float(previous[0,:2].norm())/(acceleration*dt),
                                   float(previous[0,2].abs())/(self.cfg.angular_accel*dt)))
        eligible = (r.index==2 and r.failure is None and r.pose_advanced and all(flags.values())
                    and standing and remaining >= 3.+(ramp_ticks+1)*dt)
        e.update(native_time=now, pose_native_time=r._pose_time, route_phase=r.index,
                 qualification_flags=flags, qualified_duration_s=r.qualified_duration_s,
                 pose_advanced=r.pose_advanced, prior_emitted_standing=standing,
                 remaining_timeout_s=remaining, nominal_desired_xyz=desired[0].tolist(),
                 nominal_command_xyz=nominal[0].tolist(), prior_emitted_xyz=previous[0].tolist())
        if e['state']=='armed' and eligible:
            e.update(state='ramping',trigger_native=now, admission=copy.deepcopy({k:e[k] for k in
                ('qualification_flags','pose_native_time','prior_emitted_xyz','prior_emitted_standing','remaining_timeout_s')}))
        if e['state'] in ('ramping','zero') and r.index != 2:
            self.censor_coast('route_left_prealignment', now)
        if e['state']=='zero':
            e['zero_elapsed_s'] = now-e['zero_started_native']
            if e['zero_elapsed_s'] >= 3.:
                e.update(state='completed',ended_native=now)
        override = e['state'] in ('ramping','zero')
        if override:
            delta = -previous[:,:2]
            delta *= (acceleration*dt/delta.norm(dim=-1,keepdim=True).clamp(min=1e-9)).clamp(max=1)
            self.command[:,:2] = previous[:,:2]+delta
            self.command[:,2] = previous[:,2]+(-previous[:,2]).clamp(-self.cfg.angular_accel*dt,self.cfg.angular_accel*dt)
            if e['zero_started_native'] is None and not bool(torch.count_nonzero(self.command)):
                e.update(state='zero',zero_started_native=now)
        e.update(override_desired_xyz=[0.,0.,0.] if override else None,
                 emitted_command_xyz=self.command[0].tolist())
        self._coast_last_native = now

    def _nominal_desired(self, yaw, velocity):
        cfg=self.cfg
        xy=world_xy.world_to_heading(cfg.position_gain*(self.goal-self.filtered_xy)-cfg.velocity_damping*velocity,yaw)
        xy /= (xy/xy.new_tensor([cfg.max_vx,cfg.max_vy])).norm(dim=-1,keepdim=True).clamp(min=1)
        xy[self.arrived]=0
        err=torch.atan2((self.heading-yaw).sin(),(self.heading-yaw).cos())
        wz=(cfg.heading_gain*err).clamp(-cfg.max_wz,cfg.max_wz)
        satisfied=(err.abs()<=.015)|(self.arrived&(err.abs()<=cfg.heading_tolerance))
        if not cfg.hold_heading_at_goal: satisfied |= self.arrived
        wz[satisfied]=0
        return torch.cat((xy,wz[:,None]),dim=1)

    def update(self, xy, yaw, velocity_xy_w, yaw_rate, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        selected = self.prealignment_coast_seconds is not None
        if selected:
            if self.route_context is None:
                raise ValueError('Coast requires actual reviewed route context')
            now=self.route_context._last_native
            if self._coast_last_native is not None:
                if now < self._coast_last_native: raise ValueError('Coast clock moved backward')
                if now == self._coast_last_native: return self.command
            previous=self.command.clone()
        original = self.cfg
        if self.reviewed_final_phase:
            # Match the next base-controller filter operation without mutating it.
            filtered = self.filtered_xy+(dt/(original.position_filter_s+dt))*(xy-self.filtered_xy)
            filtered[self.filter_fresh] = xy[self.filter_fresh]
            desired = world_xy.world_to_heading(
                original.position_gain*(self.goal-filtered)-original.velocity_damping*velocity_xy_w, yaw)
            if float(desired[0, 0]) > 0:
                self.cfg = replace(original, max_vx=.04)
        try:
            result=super().update(xy, yaw, velocity_xy_w, yaw_rate, dt)
            if selected:
                self._coast_update(previous,result.clone(),self._nominal_desired(yaw,velocity_xy_w),dt)
            return result
        finally:
            self.cfg = original


class ReviewedNavigationGoals(NavigationGoals):
    def __init__(self, pelvis_pose, torso_pose, payload_pose, target_payload_xyz,
                 reviewed_torso_heading, *, native_time, pose_native_time):
        super().__init__(pelvis_pose, torso_pose, payload_pose, target_payload_xyz,
                         reviewed_torso_heading)
        self.route = ReviewedApproachRoute(np.asarray(pelvis_pose)[:2], self.final_pelvis_xy,
                                           self.heading, native_time=native_time,
                                           pose_native_time=pose_native_time)
        self.goal_torso_xy = self.route.goal+self.torso_offset

    def update(self, pelvis_pose, *, native_time, pose_native_time):
        pelvis = GoalHold._pose(pelvis_pose, 'pelvis_pose')
        advanced = self.route.advance(pelvis[:2], heading=_yaw(pelvis), native_time=native_time,
                                      pose_native_time=pose_native_time)
        recovery = self.route.clearance_recovery_goal(pelvis[:2])
        self.recovering = recovery is not None
        self.goal_torso_xy = (self.route.goal if recovery is None else recovery)+self.torso_offset
        return advanced

    def apply(self, controller):
        if not isinstance(controller, ReviewedApproachController):
            raise ValueError('Explicit reviewed-approach controller required')
        super().apply(controller)
        controller.reviewed_final_phase = self.route.final
        controller.route_context = self.route
