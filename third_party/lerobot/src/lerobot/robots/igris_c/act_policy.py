"""Checkpoint-bound PJS mapping and causal interpolation; no robot I/O."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .igris_c import joint_names

OWNED = (0, 1, 2, *range(15, 29))


class PolicyContract:
    """Resolve axis order from the training dataset, never from vector length alone."""

    def __init__(self, config, info):
        self.policy_type = config.get("type")
        if self.policy_type not in ("act", "pi05", "groot") or config.get("n_obs_steps") != 1:
            raise ValueError("single-observation ACT, Pi0.5 or GR00T checkpoint required")
        if self.policy_type == "groot" and (
            config.get("embodiment_tag") != "new_embodiment"
            or not config.get("use_relative_actions")
            or config.get("chunk_size") != 16
            or config.get("n_action_steps") != 16
            or config.get("action_decode_transform") is not None
            or config.get("use_peft")
        ):
            raise ValueError("GR00T requires the 17D N1.7 relative-joint, 16-step bundle")
        if self.policy_type == "act" and config.get("n_action_steps") != 1:
            raise ValueError("rollout requires n_action_steps=1 (fresh inference each policy tick)")
        if self.policy_type == "pi05" and (
            type(config.get("chunk_size")) is not int
            or type(config.get("n_action_steps")) is not int
            or not 1 <= config["n_action_steps"] <= config["chunk_size"]
        ):
            raise ValueError("Pi0.5 requires a valid saved chunk/action horizon")
        if self.policy_type == "pi05" and (
            config.get("use_relative_actions", False)
            or config.get("rtc_config") is not None
            or config.get("use_visual_memory", False)
            or config.get("use_proprioceptive_memory", False)
        ):
            raise ValueError("Pi0.5 rollout requires absolute actions without RTC or observation memory")
        self.fps = float(info["fps"])
        if not np.isfinite(self.fps) or not 0 < self.fps <= 100:
            raise ValueError("dataset fps must be finite in (0, 100]")
        if self.policy_type == "groot" and self.fps != 15:
            raise ValueError("The GR00T IGRIS bundle was trained at 15 Hz")
        features = info["features"]
        names = {name + ".pos": i for i, name in enumerate(joint_names())}
        names.update({k.replace("left_", "l_").replace("right_", "r_"): v for k, v in names.copy().items()})
        if self.policy_type == "groot":
            # The bundle records these names AND their original 31D source indices.
            labels = [f"waist_{i}.pos" for i in range(3)] + [
                f"{side}_arm_{i}.pos" for side in ("left", "right") for i in range(7)
            ]
            for key in ("observation.state", "action"):
                if features[key]["names"] != labels or info["source_indices"][key] != list(OWNED):
                    raise ValueError("GR00T joint names/source indices differ from the IGRIS 17D bundle")
            names.update(zip(labels, OWNED, strict=True))

        def indices(key, feature):
            meta = features[key]
            labels = meta.get("names")
            if (
                not isinstance(labels, list)
                or feature["shape"] != meta["shape"]
                or len(labels) != meta["shape"][0]
            ):
                raise ValueError(f"{key}: checkpoint and dataset named position schema differ")
            try:
                result = tuple(names[name] for name in labels)
            except KeyError as exc:
                raise ValueError(f"{key}: unknown PJS joint name {exc}") from exc
            if len(set(result)) != len(result) or not result:
                raise ValueError(f"{key}: empty or duplicate joint mapping")
            return result

        inputs = config["input_features"]
        outputs = config["output_features"]
        if set(outputs) != {"action"}:
            raise ValueError("only position action is supported")
        self.state_indices = indices("observation.state", inputs["observation.state"])
        self.action_indices = indices("action", outputs["action"])
        if self.policy_type == "pi05" and config.get("action_feature_names") != features["action"]["names"]:
            raise ValueError("Pi0.5 action names must match the training dataset joint order")
        self.owned = tuple(i for i in self.action_indices if i in OWNED)
        if not set(range(15, 29)) <= set(self.owned):
            raise ValueError("both seven-axis arms are required")
        self.action_columns = tuple(self.action_indices.index(i) for i in self.owned)
        self.state_columns = tuple(self.state_indices.index(i) for i in self.owned)
        self.joint_tau_indices = ()
        if "observation.environment_state" in inputs:
            key = "observation.environment_state"
            source = "observation.joint_tau_est_Nm"
            mapping = info.get("joint_tau_input", {})
            labels = mapping.get("names")
            if (
                self.policy_type != "act"
                or inputs[key] != {"type": "ENV", "shape": [17]}
                or mapping.get("source_feature") != source
                or mapping.get("policy_feature") != key
                or mapping.get("shape") != [17]
                or mapping.get("units") != "Nm"
                or mapping.get("motor_tau_used") is not False
                or not isinstance(labels, list)
                or len(labels) != 17
                or any(
                    features.get(feature, {}).get("names") != labels
                    or features.get(feature, {}).get("shape") != [17]
                    for feature in (key, source)
                )
            ):
                raise ValueError("ACT environment_state requires the named 17D joint_tau_est_Nm contract")
            try:
                self.joint_tau_indices = tuple(names[name + ".pos"] for name in labels)
            except (KeyError, TypeError) as exc:
                raise ValueError("unknown PJS joint torque name") from exc
            if set(self.joint_tau_indices) != set(OWNED):
                raise ValueError("joint torque input must contain each waist/arm joint exactly once")
        self.images = {}
        for key, feature in inputs.items():
            if key in ("observation.state", "observation.environment_state"):
                continue
            shape = feature["shape"]
            if (
                not key.startswith("observation.images.")
                or feature["type"] != "VISUAL"
                or len(shape) != 3
                or shape[0] != 3
            ):
                raise ValueError(f"unsupported policy input: {key}")
            hwc = (shape[1], shape[2], 3)
            if any(type(n) is not int or n <= 0 for n in shape) or tuple(features[key]["shape"]) != hwc:
                raise ValueError(f"{key}: dataset image dimensions differ from checkpoint")
            self.images[key.removeprefix("observation.images.")] = hwc
        if self.policy_type == "groot":
            # Carry the live head frame intact; the saved head-only resize runs in the preprocessor.
            for key, shape in info.get("runtime_image_shapes", {}).items():
                if key != "head_cam" or key not in self.images or tuple(shape) != (720, 1280, 3):
                    raise ValueError("Unsupported GR00T live image shape")
                self.images[key] = tuple(shape)
        self.pi05_tactile_padding = tuple(info.get("pi05_tactile_padding", ()))
        if self.pi05_tactile_padding and (
            self.policy_type != "pi05"
            or self.pi05_tactile_padding != (120, 320)
            or any(self.images.get(name) != (224, 224, 3)
                   for name in ("tactile_left_heatmap", "tactile_right_heatmap"))
        ):
            raise ValueError("Pi0.5 tactile padding requires two 224x224 views and a 120x320 canvas")
        if not self.images:
            raise ValueError("at least one RGB image is required")

    def action(self, action):
        value = np.asarray(action, dtype=float)
        if value.shape != (len(self.action_indices),) or not np.isfinite(value).all():
            raise ValueError("invalid unnormalized policy action")
        return value[list(self.action_columns)].copy()

    def observation(self, q, images, *, now, state_at, max_age=0.15, joint_tau_est=None):
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (31,) or not np.isfinite(q).all() or set(images) != set(self.images):
            raise ValueError("finite 31D PJS feedback and every checkpoint image required")
        stamps = [state_at, *(stamp for _, stamp in images.values())]
        if not np.isfinite([now, *stamps]).all() or any(not 0 <= now - t <= max_age for t in stamps):
            raise TimeoutError("state or image is stale/future-dated")
        if max(stamps) - min(stamps) > 0.10:
            raise TimeoutError("state/image arrival skew exceeds 100 ms")
        result = {"observation.state": q[list(self.state_indices)].copy()}
        if self.joint_tau_indices:
            tau = np.asarray(joint_tau_est, dtype=np.float32)
            if tau.shape != (31,) or not np.isfinite(tau).all():
                raise ValueError("joint_tau_est: expected 31 finite measured LowState torque values")
            result["observation.environment_state"] = tau[list(self.joint_tau_indices)].copy()
        for key, (frame, _) in images.items():
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{key}: expected RGB uint8 HWC")
            h, w, _ = self.images[key]
            if self.policy_type == "groot":
                if frame.shape != self.images[key]:
                    raise ValueError(f"{key}: GR00T requires the uncropped source image {self.images[key]}")
                image = frame.copy()
            elif self.policy_type == "pi05":
                if self.pi05_tactile_padding and key in ("tactile_left_heatmap", "tactile_right_heatmap"):
                    if frame.shape not in ((80, 320, 3), (120, 320, 3)):
                        raise ValueError(f"{key}: expected the recorded 8-row or 12-row tactile layout")
                    if frame.shape[0] == 80:
                        frame = np.pad(frame, ((20, 20), (0, 0), (0, 0)))
                # Match the training cache: bilinear, aspect-preserving, centered black padding.
                scale = min(w / frame.shape[1], h / frame.shape[0])
                width, height = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                image = np.zeros((h, w, 3), np.uint8)
                top, left = (h - height) // 2, (w - width) // 2
                image[top : top + height, left : left + width] = resized
            else:
                image = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            result[f"observation.images.{key}"] = image
        return result


class PolicyStream:
    """Fresh policy results -> C2 quintic targets; faults require a new session."""

    def __init__(
        self,
        initial_q,
        *,
        owned=OWNED,
        fps=15,
        max_velocity=0.5,
        max_acceleration=3.0,
        max_jerk=100.0,
        max_age=0.15,
        start_tolerance=0.1,
    ):
        self.initial = np.array(initial_q, dtype=float)
        limits = [fps, max_velocity, max_acceleration, max_jerk, max_age, start_tolerance]
        if (
            self.initial.shape != (31,)
            or not np.isfinite(self.initial).all()
            or not np.isfinite(limits).all()
            or min(limits) <= 0
        ):
            raise ValueError("finite 31D start and positive stream limits required")
        self.owned = tuple(owned)
        if not self.owned or len(set(owned)) != len(owned) or not set(owned) <= set(OWNED):
            raise ValueError("only unique arm/waist axes can be owned")
        self.period = 1 / fps
        self.limits = (max_velocity, max_acceleration, max_jerk)
        self.max_age, self.start_tolerance = max_age, start_tolerance
        self.last_observed = float("-inf")
        self.pending = None
        self.fault = None
        self.lock = threading.RLock()
        self.start = None
        self.duration = 1.0
        self.coefficients = np.zeros((6, 31))
        self.coefficients[0] = self.initial

    def fail(self, reason):
        with self.lock:
            self.fault = self.fault or str(reason)

    def require_fresh(self, now):
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if not np.isfinite(now) or not 0 <= now - self.last_observed <= self.max_age:
                self.fail("policy result stale/future-dated; re-arm required")
                raise TimeoutError(self.fault)

    def submit(self, action, *, observed_at, finished_at):
        q = np.asarray(action, dtype=float)
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            try:
                if q.shape != (len(self.owned),) or not np.isfinite(q).all():
                    raise ValueError("expected finite unnormalized owned PJS action")
                if (
                    not np.isfinite([observed_at, finished_at]).all()
                    or not 0 <= finished_at - observed_at <= self.max_age
                ):
                    raise TimeoutError("policy result too old at completion")
                if observed_at <= self.last_observed:
                    raise ValueError("action source time must increase")
                if (
                    self.last_observed == float("-inf")
                    and np.max(abs(q - self.initial[list(self.owned)])) > self.start_tolerance
                ):
                    raise ValueError("first policy target too far from measured start; no automatic approach")
                self.pending = (q.copy(), observed_at)
                self.last_observed = observed_at
            except Exception as exc:
                self.fail(exc)
                raise

    def _value(self, now, derivative=0):
        u = np.clip((now - self.start) / self.duration, 0, 1)
        c = self.coefficients.copy()
        for _ in range(derivative):
            c = c[1:] * np.arange(1, len(c))[:, None] / self.duration
        if derivative and now >= self.start + self.duration:
            return np.zeros(31)
        return np.polynomial.polynomial.polyval(u, c)

    def sample(self, now):
        with self.lock:
            self.require_fresh(now)
            if self.start is None:
                self.start = now
            if self.pending is not None:
                target, observed = self.pending
                h = observed + 2 * self.period - now
                if h <= 0:
                    self.fail("expired policy interpolation endpoint")
                    raise TimeoutError(self.fault)
                q, v, a = (self._value(now, d) for d in range(3))
                goal = self.initial.copy()
                goal[list(self.owned)] = target
                c = np.zeros((6, 31))
                c[0], c[1], c[2] = q, v * h, 0.5 * a * h * h
                delta = goal - q
                c[3] = 10 * delta - 6 * c[1] - 3 * c[2]
                c[4] = -15 * delta + 8 * c[1] + 3 * c[2]
                c[5] = 6 * delta - 3 * c[1] - c[2]
                self.coefficients, self.start, self.duration = c, now, h
                self.pending = None
            for order, limit in enumerate(self.limits, 1):
                if np.max(abs(self._value(now, order)[list(self.owned)])) > limit:
                    self.fail(f"policy derivative {order} exceeds {limit}")
                    raise ValueError(self.fault)
            return self._value(now)
