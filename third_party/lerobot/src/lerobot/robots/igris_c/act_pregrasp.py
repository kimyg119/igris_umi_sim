"""Manual pre-grasp return within the existing ACT command writer."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .act_direct import polynomial_controls
from .core_trajectory import OnlineJointTrajectory
from .igris_c import joint_names

ARMS = list(range(15, 29))
UPPER_BODY = [0, 1, 2, *ARMS]


class PreGraspReset:
    def __init__(self, target, pose_file, kp, kd, *, execute, arrival_axes=None):
        raw = Path(pose_file).read_bytes()
        pose = json.loads(raw)
        self.pose_axes = UPPER_BODY if pose.get("schema") == "igris-init-pose-v1" else ARMS
        names = [joint_names()[i].replace("left_", "l_").replace("right_", "r_") for i in self.pose_axes]
        if (
            pose.get("schema") not in ("igris-pre-grasp-pose-v1", "igris-init-pose-v1")
            or pose.get("coordinate_space") != "joint"
            or pose.get("unit") != "rad"
            or pose.get("joint_names") != names
        ):
            raise ValueError("pose must contain the ordered arm or waist-and-arm PJS joints in radians")
        self.pose_q = np.asarray(pose["positions"], dtype=float)
        if self.pose_q.shape != (len(self.pose_axes),) or not np.isfinite(self.pose_q).all():
            raise ValueError(f"pose positions must be finite ({len(self.pose_axes)},)")
        self.arm_q = self.pose_q[-14:]
        if not set(self.pose_axes) <= set(target.axes):
            raise ValueError(
                "pose return requires ownership of both complete arms and any requested waist axes"
            )
        self.target = target
        self.q = target.q.copy()
        goal = self.q.copy()
        goal[self.pose_axes] = self.pose_q
        target._positions(goal)  # An exact saved pose must never be silently projected.
        self.kp, self.kd = kp.copy(), kd.copy()
        self.execute = execute
        self.arrival_axes = tuple(target.axes if arrival_axes is None else arrival_axes)
        if not set(self.arrival_axes) <= set(target.axes):
            raise ValueError("arrival axes require owned joints")
        self.phase = "policy"
        self.entered_at = 0.0
        self.arrived_since = None
        self.events = []
        self.pose_sha256 = hashlib.sha256(raw).hexdigest()
        self.curve = None

    def transition(self, phase, now):
        self.phase, self.entered_at = phase, now
        self.events.append({"phase": phase, "at": now})
        if phase == "waiting":
            label = "measured arrival" if self.execute else "shadow command preview only"
            criterion = "measured_arrival" if self.execute else "shadow_preview"
            if self.execute and not self.arrival_axes:
                label = "command trajectory complete; grasp not verified"
                criterion = "trajectory_complete"
            if getattr(self, "loaded_handoff", False):
                label = "waist measured arrival; arm command complete; grasp not verified"
                criterion = "waist_measured_arrival_arm_command_complete"
                self.events[-1]["grasp_verified"] = False
            self.events[-1].update(
                completion_criterion=criterion,
                arrival_axes=list(self.arrival_axes),
            )
            print(f"Pre-grasp: holding ({label}). Press Space to resume ACT.", flush=True)
        elif phase == "policy":
            print("Pre-grasp: ACT resumed with fresh observations.", flush=True)

    def key(self, key, pressed_at, now):
        if key == "p" and self.phase == "policy":
            # Finish the already validated segment/braking tail before moving to the saved pose.
            self.target.pending = None
            self.curve = self.target.curve
            self.transition("braking", now)
            print("Pose return: policy paused; moving directly to the saved Init/pre-grasp pose.", flush=True)
            return True
        if key == "space" and self.phase == "waiting" and pressed_at >= self.entered_at:
            self.transition("restore_gains", now)
        return False

    def plan(self, now):
        goal = self.q.copy()
        goal[self.pose_axes] = self.pose_q
        curve = OnlineJointTrajectory(self.q)
        curve.retarget(now, goal, 1.0)
        duration = 3.0
        for order, limit in enumerate(self.target.derivative_limits, 1):
            bound = np.max(abs(polynomial_controls(curve.coefficients, 1.0, order)))
            duration = max(duration, (bound / limit) ** (1 / order) * 1.01)
        curve = OnlineJointTrajectory(self.q)
        curve.retarget(now, goal, duration)
        for coefficients in (curve.coefficients, curve.stop_coefficients):
            self.target._positions(polynomial_controls(coefficients, duration))
            for order, limit in enumerate(self.target.derivative_limits, 1):
                if np.max(abs(polynomial_controls(coefficients, duration, order))) > limit + 1e-8:
                    raise ValueError("pre-grasp trajectory exceeds derivative limits")
        self.curve = curve
        self.goal = goal
        self.transition("moving", now)
        self.events[-1].update(duration_s=duration, goal=goal.tolist())

    def sample(self, now, measured):
        kp, kd = self.kp, self.kd
        if self.phase == "braking":
            self.q = self.curve.sample(now)
            kp, kd = self.kp, self.kd
            if now >= self.curve.start_s + 2 * self.curve.duration_s:
                self.transition("ready_gains", now)
        elif self.phase in ("ready_gains", "restore_gains"):
            if now - self.entered_at >= 0.5:
                if self.phase == "ready_gains":
                    self.plan(now)
                else:
                    self.transition("resuming", now)
        elif self.phase == "moving":
            self.q = self.curve.sample(now)
            if now >= self.curve.start_s + self.curve.duration_s:
                self.q = self.goal.copy()
                self.arrived_since = None
                self.transition("settling" if self.arrival_axes else "waiting", now)
        elif self.phase == "settling":
            axes = list(self.arrival_axes)
            errors = abs(np.asarray(measured)[axes] - self.q[axes])
            error = np.max(errors)
            # A wider policy start allowance must not loosen pose arrival beyond 0.2 rad.
            if not self.execute or error <= min(self.target.cfg["start_tolerance"], 0.2):
                if self.arrived_since is None:
                    self.arrived_since = now
                if now - self.arrived_since >= 0.25:
                    self.transition("waiting", now)
                    self.events[-1]["measured_error_rad"] = float(error)
            else:
                self.arrived_since = None
            if self.phase == "settling" and now - self.entered_at > 5:
                axis = axes[int(np.argmax(errors))]
                raise TimeoutError(
                    f"pre-grasp measured arrival failed: {error:.5f} rad at {joint_names()[axis]} "
                    f"(axis {axis}): measured={measured[axis]:.6f}, target={self.q[axis]:.6f}"
                )
        elif self.phase == "resuming":
            kp, kd = self.kp, self.kd
            if now - self.entered_at > 10:
                raise TimeoutError("fresh ACT result after pre-grasp timed out")
        self.target._positions(self.q)
        self.target.check_tracking(measured, self.q)
        return self.q.copy(), kp.copy(), kd.copy()
