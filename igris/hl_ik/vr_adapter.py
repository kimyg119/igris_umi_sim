"""HL-IK behind the existing VR monitor API; no CUDA IK or robot transport.

The model root is always torso. Only SDK 15:29 are optimization variables.
Observed history is sampled at 30 Hz; speculative/backoff solves never append it.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np

from igris_ik.assets import HAND_TOOL_FRAMES
from igris_ik.constants import UPPER_BODY_SDK_INDICES, URDF_JOINT_NAMES
from igris_ik.geometry import Pose, UpperBodyTargets
from igris_ik.ik import UpperBodyIK, _configuration, _orientation_error
from igris_ik.ik_config import IKConfig
from igris_ik.joint_map import ActiveJointMap
from igris_ik.result import IKDiagnostics, SolveResult
from igris_ik.robot_geometry import RobotGeometry
from igris_ik.vendor_limits import coupled_halfspaces

from .common import ARM_NAMES, OUTPUT, pose7
from .pipeline import PipelineIK
from .solver import ElbowPrior

DEFAULT_CHECKPOINT = OUTPUT / 'training/full/seed_42/best.pt'
PERIOD = 1. / 30.


class HLIKTeleopIK(UpperBodyIK):
    """Concrete CPU adapter; UpperBodyIK.__init__ (cuRobo) is intentionally unused."""

    def __init__(self, config=IKConfig(), *, checkpoint=DEFAULT_CHECKPOINT):
        self.config = replace(config, base_link='torso', waist_mode='fixed', neck_mode='fixed',
                              third_keypoint='torso', control_dt_s=PERIOD,
                              head_axes_weight=(0.,)*6, torso_axes_weight=(0.,)*6,
                              max_hand_position_error_m=.005,
                              max_hand_orientation_error_rad=np.deg2rad(2),
                              velocity_limit_rad_s={'arm': 3., 'waist': 1.5, 'neck': 3.})
        if not self.config.self_collision_check:
            raise ValueError('HL-IK teleop requires collision checking')
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'HL-IK checkpoint missing: {checkpoint}')
        self.solver = PipelineIK(ElbowPrior(checkpoint), acceleration_limit=30.)
        self._lock = threading.RLock()
        self.full_geometry = RobotGeometry(self.config.self_collision_buffer_m)
        self.hand_tool_frames = HAND_TOOL_FRAMES
        self._joint_names = ARM_NAMES
        self.joint_map = ActiveJointMap.from_names(ARM_NAMES)
        bounds = np.array([[self.full_geometry.urdf.joint_map[n].limit.lower,
                            self.full_geometry.urdf.joint_map[n].limit.upper]
                           for n in URDF_JOINT_NAMES]).T
        for name, limits in self.config.waist_limits_deg.items():
            i = URDF_JOINT_NAMES.index(name)
            bounds[:, i] = [max(bounds[0, i], np.deg2rad(limits[0])),
                            min(bounds[1, i], np.deg2rad(limits[1]))]
        self._bounds = bounds
        # These are the three model fields consumed by RobotMonitor/VR feedback.
        self.model = SimpleNamespace(active_joint_names=ARM_NAMES, _joint_offsets=np.zeros(31),
                                     full_joint_limits=lambda: tuple(self._bounds.copy()))
        self._a, self._b = coupled_halfspaces(list(URDF_JOINT_NAMES))
        self._previous = None
        self._observed = None
        self._next_history_at = None
        self.history_updates = self.history_resets = self.solve_count = 0
        self.last_pipeline_result = None
        self.reset(self.default_configuration())

    def reset(self, q_sdk=None):
        with self._lock:
            q = _configuration(self.default_configuration() if q_sdk is None else q_sdk).q_sdk
            self.solver.reset(self._solver_q(q))
            self._previous = q.copy()
            self._observed = None
            self._next_history_at = None

    def _solver_q(self, q):
        physical = np.asarray(q) + self.model._joint_offsets
        # Training MJCF bakes neck=0. Neck cannot affect torso-root arm FK/prior;
        # full_geometry checks the ACTUAL neck/head separately before acceptance.
        physical[29:] = 0.
        return physical

    def warmup(self, q_sdk):
        self.reset(q_sdk)
        self.solver.prior.predict(np.array(self.solver.history), self.solver.states()[:, :7])

    def lock_to(self, q_sdk, *, preserve_warm_start=False, joint_offsets=None):
        with self._lock:
            if joint_offsets is not None:
                offsets = np.asarray(joint_offsets, float)
                if offsets.shape != (31,) or not np.isfinite(offsets).all():
                    raise ValueError('joint_offsets must contain 31 finite angles')
                self.model._joint_offsets = offsets.copy()
            if not preserve_warm_start:
                self.reset(q_sdk)

    def observe(self, q_sdk, now):
        """Interpolate measurements onto the trained 30 Hz clock, never command history.

        A >100 ms observation gap resets the five-frame warmup rather than fabricating
        motion across missing input. No network/history update occurs during IK retries.
        """
        with self._lock:
            q = self._solver_q(_configuration(q_sdk).q_sdk)
            now = float(now)
            if not np.isfinite(now):
                raise ValueError('finite observation time required')
            if self._observed is None or not 0 < now-self._observed[0] <= .1:
                self.solver.reset(q)
                self._next_history_at = now + PERIOD
                self.history_resets += 1
            else:
                stamp, old = self._observed
                while self._next_history_at <= now + 1e-9:
                    alpha = np.clip((self._next_history_at-stamp)/(now-stamp), 0., 1.)
                    self.solver.set_q(old + alpha*(q-old))
                    self.solver.history.append(self.solver.states())
                    self._next_history_at += PERIOD
                    self.history_updates += 1
            self._observed = (now, q.copy())

    def warm_start(self):
        with self._lock:
            return (self._previous.copy(), self.solver.velocity.copy(), self.solver.acceleration.copy())

    def restore_warm_start(self, snapshot):
        with self._lock:
            self._previous, self.solver.velocity, self.solver.acceleration = [v.copy() for v in snapshot]

    def forward(self, q_sdk):
        with self._lock:
            q = _configuration(q_sdk).q_sdk + self.model._joint_offsets
            self.full_geometry.update(q)
            poses = {}
            for key, name in zip(('left', 'right', 'head', 'torso'), (*HAND_TOOL_FRAMES, 'head', 'torso')):
                p = pose7(self.full_geometry.urdf.get_transform(name, 'torso'))
                poses[key] = Pose(p[:3], p[3:])
            return UpperBodyTargets(**poses)

    def _static_constraints(self, q_sdk):
        q = _configuration(q_sdk).q_sdk
        reasons = []
        if (np.any(q[UPPER_BODY_SDK_INDICES] < self._bounds[0, UPPER_BODY_SDK_INDICES]-1e-8)
                or np.any(q[UPPER_BODY_SDK_INDICES] > self._bounds[1, UPPER_BODY_SDK_INDICES]+1e-8)
                or np.any(self._a @ q > self._b+1e-8)):
            reasons.append('joint_limits')
        physical = q + self.model._joint_offsets
        self.solver.set_q(self._solver_q(q))
        if self.solver.geometry.collision(self.solver.configuration.data):
            reasons.append('self_collision')
        reason = self.full_geometry.command_guard(physical, physical)
        if reason:
            reasons.append(reason)
        return tuple(reasons)

    def check_configuration(self, q_sdk, targets, *, update_warm_start=False):
        with self._lock:
            started = time.perf_counter()
            q = _configuration(q_sdk)
            constraints = self._static_constraints(q)
            actual = self.forward(q)
            errors = [(float(np.linalg.norm(getattr(actual, side).position-getattr(targets, side).position)),
                       _orientation_error(getattr(actual, side).wxyz, getattr(targets, side).wxyz))
                      for side in ('left', 'right')]
            reasons = list(constraints)
            if max(e[0] for e in errors) > self.config.max_hand_position_error_m:
                reasons.append('hand_position_residual')
            if max(e[1] for e in errors) > self.config.max_hand_orientation_error_rad:
                reasons.append('hand_orientation_residual')
            diagnostics = IKDiagnostics(time.perf_counter()-started, errors[0][0], errors[1][0],
                                        errors[0][1], errors[1][1], 0., False)
            result = SolveResult(q, not reasons, not any('collision' in n for n in constraints),
                                 tuple(reasons), diagnostics, constraints_satisfied=not constraints)
            if update_warm_start and result.success:
                self.reset(q)
            return result

    def solve(self, q_sdk, targets, *, elbow_targets=None):
        if elbow_targets is not None:
            raise ValueError('HL-IK uses its learned elbow prior; use --arm-tracking off')
        with self._lock:
            started = time.perf_counter()
            before = self.warm_start()
            seed = _configuration(q_sdk).q_sdk.copy()
            seed[15:29] = self._previous[15:29]
            constraints = self._static_constraints(seed)
            if constraints:
                return self.check_configuration(seed, targets)
            goals = np.array([np.r_[p.position, p.wxyz] for p in (targets.left, targets.right)])
            raw = self.solver.solve(goals, self._solver_q(seed), record_history=False,
                                    dt=PERIOD, velocity_limit=3.)
            self.solve_count += 1
            self.last_pipeline_result = raw
            candidate = seed.copy()
            candidate[15:29] = raw.q[15:29]-self.model._joint_offsets[15:29]
            result = self.check_configuration(candidate, targets)
            # An accepted QP tick is not necessarily a converged VR waypoint.
            if not raw.accepted or raw.prior_status != 'used':
                reason = raw.reason if not raw.accepted else 'hlik_prior_invalid'
                result = replace(result, success=False, constraints_satisfied=False,
                                 reasons=(*result.reasons, reason))
            if result.success:
                self._previous = candidate.copy()
            else:
                self.restore_warm_start(before)
            return replace(result, diagnostics=replace(result.diagnostics,
                           solve_time_s=time.perf_counter()-started))
