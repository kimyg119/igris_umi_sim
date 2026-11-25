"""Fixed-rate latest-action sender, independent of ACT inference and igris_core."""

import copy
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .core_safe_set import SafeSetPolygon
from .core_trajectory import OnlineJointTrajectory
from .direct_dds import vector
from .igris_c import joint_names

PERIOD = 0.010
MAX_LATENESS = 0.010
MAX_AGE = 0.150
ELBOW_LOWER_BACKOFF_RAD = 0.02
PARALLEL = (0, 1, 2, 7, 8, 13, 14, 19, 20, 21, 26, 27, 28)
DERIVATIVE_GATES = ("velocity_limit_gate", "acceleration_limit_gate", "jerk_limit_gate")


def selected_axes(contract, selection):
    axes = tuple(
        i for i in contract.owned if selection == "model" or (i >= 15 if selection == "arms" else i < 3)
    )
    if not axes:
        raise ValueError("no model outputs match selected joints")
    return axes


# Bernstein control points bound the whole polynomial, including between DDS ticks.
def _bound_matrix(degree):
    bernstein = np.array(
        [
            [math.comb(k, i) / math.comb(degree, i) if i <= k else 0 for i in range(degree + 1)]
            for k in range(degree + 1)
        ]
    )
    return np.array(
        [
            bernstein
            @ np.array(
                [
                    [
                        math.comb(i, j) * (part / 4) ** (i - j) / 4**j if i >= j else 0
                        for i in range(degree + 1)
                    ]
                    for j in range(degree + 1)
                ]
            )
            for part in range(4)
        ]
    )


_BOUND = {degree: _bound_matrix(degree) for degree in range(4, 8)}


def polynomial_controls(coefficients, duration, derivative=0):
    c = coefficients
    for _ in range(derivative):
        c = c[1:] * np.arange(1, len(c))[:, None] / duration
    return _BOUND[len(c) - 1] @ c


def upper_body_polygons(axes):
    polygons = []
    for group, pair in (("waist", (1, 2)), ("wrist_L", (20, 21)), ("wrist_R", (27, 28))):
        if not set(pair) & set(axes):
            continue
        path = Path(__file__).with_name("safe_set_polygons") / f"{group}_joint_safe_polygon.json"
        payload = json.loads(path.read_text())
        polygon = SafeSetPolygon(
            group, group, "joint", *pair, tuple(tuple(v) for v in payload["vertices"]), str(path), 0.0
        )
        polygons.append(polygon.scaled(0.95))
    return polygons


class InterpolatedTarget:
    """Ensembled low-rate positions -> continuous C3 joint targets at wall-clock time."""

    def __init__(self, initial, contract, axes, limits, cfg):
        self.initial = vector(initial, "initial q").copy()
        self.q = self.initial.copy()
        self.goal = self.initial.copy()
        self.requested_goal = self.initial.copy()
        # Deprecated compatibility indicator: 1 = unchanged, 0 = conditioned.
        # Group projection has no common trajectory progress fraction.
        self.projection_fraction = 1.0
        self.target_projected = False
        self.correction_axes = []
        self.max_correction_rad = 0.0
        self.correction_warnings = []
        self.tracking_error_rad = 0.0
        self.tracking_error_axis = None
        self.axes = list(axes)
        self.columns = [contract.owned.index(i) for i in axes]
        self.dimension = len(contract.owned)
        self.period = 1 / contract.fps
        self.lower, self.upper = limits
        self.cfg = cfg
        self.seq = -1
        self.observed = float("-inf")
        self.pending = None
        self.curve = OnlineJointTrajectory(initial)
        self.polygons = upper_body_polygons(axes)
        # Leave room for polynomial roundoff while retaining the original trajectory guards.
        self.projection_polygons = [polygon.scaled(1.0 - 1e-5) for polygon in self.polygons]
        self.derivative_limits = [cfg[k] for k in ("max_velocity", "max_acceleration", "max_jerk")]
        self.last_duration = None
        self.planned_endpoint = None
        self.holding = False
        self.hold_events = 0
        self.stale_results_dropped = 0
        self.replan_rejections = 0
        self.last_replan_error = None
        self.applied_seq = -1
        self.applied_at = None
        self.replan_status = "waiting"

    def diagnostics(self):
        goal_gap = None if self.planned_endpoint is None else float(
            np.max(abs(self.goal[self.axes] - self.planned_endpoint[0][self.axes]))
        )
        return {"policy_seq": self.seq, "observed": self.observed if self.seq >= 0 else None,
                "applied_seq": self.applied_seq, "applied_at": self.applied_at,
                "replan_status": self.replan_status, "replan_rejections": self.replan_rejections,
                "last_replan_error": self.last_replan_error, "goal_gap_rad": goal_gap,
                "holding": self.holding, "stale_results_dropped": self.stale_results_dropped,
                "entry_until": getattr(self, "entry_until", None),
                "projection_mode": "axis_group_minimum" if self.cfg.get("boundary_policy") == "project" else "reject",
                "target_projected": self.target_projected,
                "correction_axes": list(self.correction_axes),
                "max_correction_rad": self.max_correction_rad,
                "correction_warnings": [dict(warning) for warning in self.correction_warnings],
                **{name: self.cfg.get(name, True) for name in DERIVATIVE_GATES}}

    def derivative_limit_enabled(self, order):
        return self.cfg.get(DERIVATIVE_GATES[order - 1], True)

    def _positions(self, points, *, projection=False):
        margin = 1e-6 if projection else 0.0
        if np.any(points[..., self.axes] < self.lower[self.axes] + margin - 1e-9) or np.any(
            points[..., self.axes] > self.upper[self.axes] - margin + 1e-9
        ):
            raise ValueError("interpolated target outside joint limits")
        for point in np.reshape(points, (-1, 31)):
            for polygon in self.projection_polygons if projection else self.polygons:
                if not polygon.contains((point[polygon.x_axis], point[polygon.y_axis])):
                    raise ValueError(f"interpolated target outside {polygon.name} coupled safe set")

    def _prepare_goal(self, requested):
        project = self.cfg.get("boundary_policy", "reject") == "project"
        try:
            self._positions(requested, projection=project)
            return requested, 1.0
        except ValueError:
            if not project:
                raise
        self._positions(self.q)
        maximum = float(self.cfg.get("max_projection_correction_rad", 0.01))
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("max_projection_correction_rad must be finite and positive")
        goal = requested.copy()
        lower, upper = self.lower + 1e-6, self.upper - 1e-6
        if np.any(lower[self.axes] > upper[self.axes]):
            raise ValueError("joint limits have no projection interior")
        goal[self.axes] = np.clip(goal[self.axes], lower[self.axes], upper[self.axes])
        backoff_elbows = [
            axis for axis in (18, 25)
            if axis in self.axes and requested[axis] < lower[axis] - 1e-9
        ]
        for axis in backoff_elbows:
            goal[axis] = self.lower[axis] + ELBOW_LOWER_BACKOFF_RAD
        for polygon in self.projection_polygons:
            pair = [polygon.x_axis, polygon.y_axis]
            if not polygon.contains(goal[pair]):
                goal[pair] = polygon.project(
                    requested[pair], lower[pair], upper[pair],
                    movable=[axis in self.axes for axis in pair],
                )
        self._positions(goal, projection=True)
        return goal, 0.0

    def _correction_warnings(self, requested, goal):
        threshold = float(self.cfg.get("max_projection_correction_rad", 0.01))
        names = joint_names()
        warnings = []
        for axis in self.axes:
            axis_threshold = threshold
            if axis in (18, 25) and requested[axis] < self.lower[axis] + 1e-6 - 1e-9:
                axis_threshold += ELBOW_LOWER_BACKOFF_RAD
            correction = float(abs(goal[axis] - requested[axis]))
            if correction > axis_threshold:
                warnings.append({"axis": axis, "name": names[axis],
                                 "requested": float(requested[axis]), "corrected": float(goal[axis]),
                                 "correction_rad": correction, "threshold_rad": axis_threshold})
        return warnings

    def check_start(self, goal, measured):
        measured = vector(measured, "measured")
        differences = abs(goal[self.axes] - measured[self.axes])
        worst = int(np.argmax(differences))
        delta = differences[worst]
        if delta > self.cfg["start_tolerance"]:
            axis = self.axes[worst]
            raise ValueError(
                f"start mismatch {delta:.5f} rad ({np.degrees(delta):.2f} deg) "
                f"at {joint_names()[axis]} (axis {axis}): "
                f"measured={measured[axis]:.6f} rad, target={goal[axis]:.6f} rad, "
                f"tolerance={self.cfg['start_tolerance']:.5f} rad; "
                "position robot manually, no auto approach"
            )

    def submit(self, seq, observed, finished, action, measured, now):
        action = np.asarray(action, float)
        if action.shape != (self.dimension,) or not np.isfinite(action).all():
            raise ValueError("invalid unnormalized model action")
        if seq <= self.seq or observed <= self.observed or not observed <= finished <= now:
            raise ValueError("nonmonotonic policy result")
        if not 0 <= now - observed <= MAX_AGE:
            raise TimeoutError("policy result stale")
        goal = self.initial.copy()
        goal[self.axes] = action[self.columns]
        requested = goal.copy()
        # Projection must not hide a first-frame mismatch or become an auto-approach.
        if self.seq < 0 and self.cfg.get("boundary_policy") == "project":
            self.check_start(requested, measured)
        goal, fraction = self._prepare_goal(requested)
        velocity = np.zeros(31)
        if self.seq < 0:
            self.check_start(requested, measured)
        else:
            delta = goal[self.axes] - self.goal[self.axes]
            step_magnitudes = abs(delta)
            if self.cfg.get("waist_yaw_step_gate", True) is False:
                step_magnitudes[np.asarray(self.axes) == 0] = 0.0
            worst = int(np.argmax(step_magnitudes))
            if step_magnitudes[worst] > self.cfg["max_action_step"]:
                axis = self.axes[worst]
                names = joint_names()
                exceeded = ", ".join(
                    f"{names[i]}(axis {i}):{abs(step):.6f}rad"
                    for i, step in zip(self.axes, step_magnitudes, strict=True)
                    if abs(step) > self.cfg["max_action_step"]
                )
                raise ValueError(
                    f"policy target step exceeds max_action_step: {names[axis]} (axis {axis}), "
                    f"step={delta[worst]:+.6f} rad (abs={abs(delta[worst]):.6f} rad, "
                    f"{np.degrees(abs(delta[worst])):.2f} deg), limit={self.cfg['max_action_step']:.6f} rad; "
                    f"previous={self.goal[axis]:.6f}, target={goal[axis]:.6f}, "
                    f"requested={requested[axis]:.6f} rad; "
                    f"seq={self.seq}->{seq} (gap={seq - self.seq}), "
                    f"observation_interval_ms={(observed - self.observed) * 1000:.3f}, "
                    f"observation_age_ms={(now - observed) * 1000:.3f}, "
                    f"result_latency_ms={(finished - observed) * 1000:.3f}; "
                    f"projection={self.projection_fraction:.6f}->{fraction:.6f}; exceeded={exceeded}"
                )
            if not self.holding and self.cfg.get("endpoint_velocity", "difference") == "difference":
                velocity[self.axes] = delta / (observed - self.observed)
        self.pending = (goal, velocity, observed)
        self.goal = goal
        self.requested_goal = requested
        self.projection_fraction = fraction
        correction = abs(goal - requested)
        self.correction_axes = [axis for axis in self.axes if correction[axis] > 0]
        self.max_correction_rad = float(np.max(correction[self.axes]))
        self.target_projected = bool(self.correction_axes)
        self.correction_warnings = self._correction_warnings(requested, goal) if self.target_projected else []
        self.seq, self.observed = seq, observed
        self.holding = False

    def rebase_before_arm(self, initial):
        self.initial = vector(initial, "initial q").copy()
        self.curve = OnlineJointTrajectory(initial)
        self.q = self.initial.copy()
        self.planned_endpoint = None
        unowned = [i for i in range(31) if i not in self.axes]
        self.goal[unowned] = self.initial[unowned]
        self.requested_goal[unowned] = self.initial[unowned]
        if self.pending:
            self.pending[0][unowned] = self.initial[unowned]

    def check(self, now, measured, q):
        age = now - self.observed
        can_hold = self.cfg.get("policy_stale_behavior", "stop") == "hold" and self.seq >= 0
        if not np.isfinite(age) or age < 0 or (age > MAX_AGE and not can_hold):
            raise TimeoutError(f"policy result stale; sender stopped (age={age:.3f}s, seq={self.seq})")
        self.check_tracking(measured, q)

    def check_tracking(self, measured, q):
        measured = vector(measured, "measured")
        differences = abs(q[self.axes] - measured[self.axes])
        worst = int(np.argmax(differences))
        delta = differences[worst]
        self.tracking_error_rad = float(delta)
        self.tracking_error_axis = axis = self.axes[worst]
        if self.cfg.get("tracking_error_gate", True) and delta > self.cfg["max_tracking_error"]:
            raise ValueError(
                f"owned joint tracking error {delta:.5f} rad ({np.degrees(delta):.2f} deg) "
                f"at {joint_names()[axis]} (axis {axis}): "
                f"command={q[axis]:.6f} rad, measured={measured[axis]:.6f} rad, "
                f"limit={self.cfg['max_tracking_error']:.5f} rad"
            )

    def sample(self, now, measured):
        self.check(now, measured, self.q)
        if now - self.observed > MAX_AGE:
            if not self.holding:
                # Freeze the last sampled command, not the old trajectory's future endpoint.
                self.curve = OnlineJointTrajectory(self.q)
                self.pending = None
                self.planned_endpoint = None
                self.holding = True
                self.hold_events += 1
                self.replan_status = "stale_hold"
            self._positions(self.q)
            return self.q.copy()
        if self.pending is not None:
            goal, velocity, observed = self.pending
            # Replanning an unchanged endpoint can overshoot; its validated curve already reaches it.
            if self.planned_endpoint is not None and all(
                np.array_equal(a, b) for a, b in zip((goal, velocity), self.planned_endpoint, strict=True)
            ):
                self.pending = None
                self.replan_status = "unchanged_goal"
        if self.pending is not None:
            goal, velocity, observed = self.pending
            duration = max(2 * self.period, observed + self.cfg["interpolation_delay"] - now)
            candidate = copy.copy(self.curve)
            candidate.retarget(now, goal, duration, velocity=velocity)
            try:
                for coefficients in (candidate.coefficients, candidate.stop_coefficients):
                    self._positions(polynomial_controls(coefficients, duration))
                    for order, limit in enumerate(self.derivative_limits, 1):
                        if not self.derivative_limit_enabled(order):
                            continue
                        bound = np.max(
                            abs(polynomial_controls(coefficients, duration, order)[..., self.axes])
                        )
                        if bound > limit + 1e-8:
                            raise ValueError(
                                f"interpolation derivative {order} bound {bound:.4f} exceeds {limit}; increase interpolation_delay or review target"
                            )
            except ValueError as exc:
                if self.cfg.get("boundary_policy") != "project" or self.planned_endpoint is None:
                    raise
                # Keep the validated segment and braking tail; retry only with a fresh policy result.
                self.replan_rejections += 1
                self.last_replan_error = str(exc)
                self.replan_status = "rejected"
            else:
                self.curve = candidate
                self.planned_endpoint = (goal.copy(), velocity.copy())
                self.last_duration = duration
                self.applied_seq = self.seq
                self.applied_at = now
                self.replan_status = "applied"
            self.pending = None
        q = self.curve.sample(now)
        self._positions(q)
        self.check(now, measured, q)
        self.q = q
        return q.copy()


def require_deadline(now, deadline, *, gate=True):
    if gate and now - deadline > MAX_LATENESS:
        raise TimeoutError(f"100 Hz deadline missed by {(now - deadline) * 1000:.2f} ms; no catch-up")
    return max(0.0, now - deadline)


def next_deadline(deadline, now, *, gate=True):
    # A relaxed overrun starts a new 10 ms period instead of bursting missed commands.
    if not gate and now - deadline > MAX_LATENESS:
        return now + PERIOD
    return deadline + PERIOD


def write_feedback(shared, q, stamp, joint_tau_est=None):
    values = [*q, stamp]
    if joint_tau_est is not None:
        values.extend(joint_tau_est)
    with shared.get_lock():
        shared[:] = values


class Feedback:
    def __init__(self, shared):
        self.shared = shared

    @staticmethod
    def clock():
        return time.monotonic()

    def snapshot(self):
        with self.shared.get_lock():
            values = list(self.shared[:])
        return {
            "joint_q": values[:31],
            "joint_tau_est": values[32:],
            "ages_s": {"lowstate": time.monotonic() - values[31]},
        }


def inference_worker(
    cfg,
    shared,
    pipe,
    stop,
    recording_frames=None,
    command_shared=None,
    control_generation=None,
    preview=None,
    diagnostics_path=None,
    policy_switches=None,
):
    # Camera IO and local/remote inference stay outside the fixed-rate sender.
    import queue

    from .act_runtime import ActInference, make_sensors, observe
    from .act_diagnostics import InferenceDiagnostics

    if recording_frames is not None:
        import signal

        from .act_recording import recording_frame

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with ExitStack() as resources:
            diagnostics = InferenceDiagnostics()
            diagnostics_log = resources.enter_context(Path(diagnostics_path).open("x", buffering=1)) if diagnostics_path else None
            if cfg.get("inference_server"):
                from .act_server import RemoteActInference

                inference = RemoteActInference(cfg)
                resources.callback(lambda: inference.close())
                print(f"Connected to resident ACT (PID {inference.server_pid})", flush=True)
            else:
                import torch

                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
                info = json.loads(Path(cfg["dataset_info"]).read_text())
                inference = ActInference(
                    Path(cfg["checkpoint"]),
                    info,
                    cfg["device"],
                    temporal_ensemble_coeff=cfg["temporal_ensemble_coeff"],
                    head_cam_attention=cfg.get("head_cam_attention", False),
                    tactile_attention=cfg.get("tactile_attention", False),
                    task=cfg.get("dataset_task"),
                )
            monitor_images = {}
            sensor_diagnostics = {}
            if cfg["input_mode"] == "live":
                sensors = (
                    Path(cfg["sensors"])
                    if cfg["sensors"]
                    else {k: cfg.get(k) for k in ("cameras", "tactile", "ros2_bundle")}
                )
                cameras, tactile, bundle = make_sensors(sensors, inference.contract)
                for sensor in [bundle] if bundle is not None else [*cameras.values(), *tactile.values()]:
                    resources.callback(sensor.disconnect)
                    sensor.connect()

                def read():
                    observation, stamp = observe(
                        Feedback(shared),
                        cameras,
                        tactile,
                        inference.contract,
                        bundle=bundle,
                        include_joint_torque=recording_frames is not None,
                        preview_images=monitor_images if preview is not None else None,
                        diagnostics=sensor_diagnostics,
                    )
                    # Only the full recording array is removed; checkpoint ENV torque stays in the input.
                    joint_tau_est = observation.pop("observation.joint_tau_est_Nm", None)
                    return observation, stamp, joint_tau_est
            else:
                from .act_server import validate_observation

                with np.load(cfg["sample"], allow_pickle=False) as archive:
                    sample = dict(archive)
                validate_observation(sample, inference.contract)

                def read():
                    stamp = time.monotonic()
                    sensor_diagnostics.clear()
                    sensor_diagnostics.update(mode="sample", sensors={
                        name: {"kind": "tactile" if "tactile" in name else "camera",
                               "source": "fixed_sample", "frame_id": "sample:0", "received_at": None}
                        for name in inference.contract.images
                    })
                    return sample, stamp, None

            if cfg.get("init_pose_on_start"):
                read()  # Verify fresh sensors before allowing the initial physical approach.
                pipe.send(("ready",))

            active_generation = None
            last_control_reset = 0.0
            warmed_up = 0

            def switch_policy(requested):
                nonlocal inference, cfg, diagnostics, last_control_reset, sample, warmed_up
                if policy_switches is None:
                    return
                try:
                    switch_id, candidate_cfg = policy_switches.get_nowait()
                except queue.Empty:
                    return
                candidate = None
                try:
                    if not cfg.get("inference_server") or not candidate_cfg.get("inference_server"):
                        raise ValueError("in-place switching requires resident policies")
                    if candidate_cfg["inference_server"] == cfg["inference_server"]:
                        raise ValueError("requested resident policy is already selected")
                    for name in ("input_mode", "sensors", "cameras", "tactile", "ros2_bundle",
                                 "policy_stale_behavior"):
                        if candidate_cfg.get(name) != cfg.get(name):
                            raise ValueError(f"policy switch changes live sensor/control setting: {name}")
                    candidate = RemoteActInference(candidate_cfg)
                    for name in ("policy_type", "fps", "state_indices", "action_indices", "owned",
                                 "action_columns", "state_columns", "joint_tau_indices", "images",
                                 "pi05_tactile_padding"):
                        if getattr(candidate.contract, name) != getattr(inference.contract, name):
                            raise ValueError(f"policy switch changes the observation/action contract: {name}")
                    candidate_sample = None
                    if cfg["input_mode"] != "live":
                        with np.load(candidate_cfg["sample"], allow_pickle=False) as archive:
                            candidate_sample = dict(archive)
                        validate_observation(candidate_sample, candidate.contract)
                    candidate.reset()
                    if stop.wait(0) or int(control_generation.value) != requested or not requested % 2:
                        raise RuntimeError("policy switch requires an unchanged paused generation")
                except Exception as exc:
                    if candidate is not None:
                        candidate.close()
                    pipe.send(("switch_error", switch_id, f"{type(exc).__name__}: {exc}"))
                    return
                previous, inference = inference, candidate
                cfg = dict(candidate_cfg)
                if candidate_sample is not None:
                    sample = candidate_sample
                diagnostics = InferenceDiagnostics()
                sensor_diagnostics.clear()
                monitor_images.clear()
                warmed_up = 0
                last_control_reset = time.monotonic()
                previous.close()
                pipe.send(("switched", switch_id))

            def wait_running():
                nonlocal active_generation, last_control_reset
                if control_generation is None:
                    return None
                if stop.wait(0):
                    return -1
                while True:
                    requested = int(control_generation.value)
                    if requested != active_generation:
                        # A new generation must never reuse predictions from the previous episode.
                        inference.reset()
                        active_generation = requested
                        last_control_reset = time.monotonic()
                        if requested % 2:
                            pipe.send(("paused", requested))
                        # A reset RPC can overlap another manual-control transition.
                        continue
                    if not requested % 2:
                        return requested
                    switch_policy(requested)
                    if time.monotonic() - last_control_reset >= 5.0:
                        # The resident server disconnects a client after 30 seconds without an RPC.
                        inference.reset()
                        last_control_reset = time.monotonic()
                    if stop.wait(0.05):
                        return -1

            def still_running(generation):
                return control_generation is None or int(control_generation.value) == generation

            while warmed_up < 8:
                generation = wait_running()
                if generation == -1:
                    return
                observation, stamp, _ = read()
                if not still_running(generation):
                    continue
                if generation is not None and generation > 0 and stamp < last_control_reset:
                    continue
                inference(observation)
                if still_running(generation):
                    warmed_up += 1
            inference.reset()
            deadline = time.monotonic()
            loop_generation = active_generation
            seq = 0
            period = 1 / inference.contract.fps
            hold = cfg.get("policy_stale_behavior", "stop") == "hold"
            while not stop.wait(max(0, deadline - time.monotonic())):
                generation = wait_running()
                if generation == -1:
                    return
                if generation != loop_generation:
                    # An intentional pause does not count as a missed inference deadline.
                    deadline = time.monotonic()
                    loop_generation = generation
                if time.monotonic() - deadline > period:
                    if not hold:
                        raise TimeoutError("inference scheduling deadline missed")
                    # Skipped policy ticks invalidate temporal-ensemble time alignment.
                    inference.reset()
                    deadline = time.monotonic()
                observation, stamp, joint_tau_est = read()
                if command_shared is not None:
                    with command_shared.get_lock():
                        command = list(command_shared[:])
                if not still_running(generation):
                    continue
                if generation is not None and generation > 0 and stamp < last_control_reset:
                    # Resume only from samples received after the ensemble reset completed.
                    deadline = time.monotonic()
                    continue
                started = time.monotonic()
                action = inference(observation)
                finished = time.monotonic()
                model_diagnostics = getattr(inference, "last_diagnostics", {})
                model_diagnostics = dict(model_diagnostics) if isinstance(model_diagnostics, dict) else {}
                attention_maps = (
                    inference.attention_maps
                    if cfg.get("head_cam_attention") or cfg.get("tactile_attention")
                    else None
                )
                if not still_running(generation):
                    continue
                if finished - stamp > MAX_AGE:
                    if not hold:
                        raise TimeoutError("observation/inference exceeded freshness limit")
                    inference.reset()
                # The receiver discards expired results, including delays incurred in IPC.
                message = ("action", seq, stamp, finished, action.tolist(), finished - started)
                if control_generation is not None:
                    message += (generation,)
                pipe.send(message)
                diagnostic = diagnostics.update(seq=seq, generation=generation, observed=stamp,
                    started=started, finished=finished, action=action,
                    sensors=sensor_diagnostics, inference=model_diagnostics)
                if diagnostics_log is not None:
                    diagnostics_log.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                if preview is not None and still_running(generation):
                    preview.publish(observation | monitor_images, seq, stamp, finished,
                                    started=started, diagnostics=diagnostic)
                if recording_frames is not None and still_running(generation):
                    frame = recording_frame(
                        inference.contract,
                        observation,
                        action,
                        command,
                        stamp=stamp,
                        started=started,
                        finished=finished,
                        seq=seq,
                        task=cfg["dataset_task"],
                        joint_tau_est=joint_tau_est,
                        attention_maps=attention_maps,
                        policy={
                            "id": cfg["recording_policy_id"],
                            "name": cfg["recording_policy_name"],
                            "checkpoint": inference.identity["checkpoint"],
                            "model_sha256": inference.identity["files"]["model.safetensors"],
                        } if cfg.get("annotate_policy") else None,
                    )
                    if frame is not None and still_running(generation):
                        # Overload is an explicit error, never a silently dropped training frame.
                        try:
                            recording_frames.put_nowait(frame)
                        except queue.Full:
                            raise RuntimeError(
                                "rollout recording/display queue full; recording cannot keep up"
                            ) from None
                seq += 1
                deadline += period
    except BaseException as exc:
        pipe.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        pipe.close()
        if recording_frames is not None:
            recording_frames.close()
            recording_frames.join_thread()


def receive_latest(pipe, worker, target, measured, diagnostics, *, generation=None, on_control=None):
    latest = None
    while pipe.poll():
        message = pipe.recv()
        if message[0] == "error":
            raise RuntimeError(message[1])
        if message[0] in ("paused", "ready"):
            continue
        if message[0] in ("switched", "switch_error"):
            if on_control is not None:
                on_control(message)
            continue
        if generation is not None and (len(message) != 7 or message[6] != generation):
            continue
        latest = message
    if not worker.is_alive():
        raise RuntimeError(f"inference worker exited: {worker.exitcode}")
    if latest:
        _, seq, stamp, finished, action, duration = latest[:6]
        now = time.monotonic()
        if target.cfg.get("policy_stale_behavior", "stop") == "hold" and now - stamp > MAX_AGE:
            target.stale_results_dropped += 1
            return False
        try:
            target.submit(seq, stamp, finished, action, measured, now)
        except ValueError as exc:
            raise ValueError(f"{exc}; inference_ms={duration * 1000:.3f}") from exc
        diagnostics.append(duration)
    return latest is not None
