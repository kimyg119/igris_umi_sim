"""Observation, model, and trajectory diagnostics independent of image previews."""

import copy
import json
import math
import time
from collections import deque

import numpy as np


WINDOW_S = 2.0
STALE_S = 0.150
SAME_ACTION_ATOL = 1e-6


class DiagnosticChannel:
    """Bounded process-shared JSON; diagnostics never wait on an image encoder."""

    def __init__(self, ctx):
        self.buffer = ctx.RawArray("B", 65536)
        self.length = ctx.RawValue("i", 0)
        self.lock = ctx.Lock()

    def publish(self, value):
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(payload) > len(self.buffer) or not self.lock.acquire(False):
            return False
        try:
            self.buffer[:len(payload)] = payload
            self.length.value = len(payload)
        finally:
            self.lock.release()
        return True

    def read(self):
        if not self.lock.acquire(timeout=0.005):
            return None
        try:
            size = self.length.value
            return json.loads(bytes(self.buffer[:size])) if size else None
        finally:
            self.lock.release()


def _events(history, now):
    while history and history[0] < now - WINDOW_S:
        history.popleft()
    return list(history)


def _age(stamp, now):
    if stamp is None or not math.isfinite(stamp) or stamp > now:
        return None
    return 1000 * (now - stamp)


def _hz(events, now):
    if events is None:
        return None
    return sum(now - WINDOW_S <= stamp <= now for stamp in events) / WINDOW_S


class InferenceDiagnostics:
    """Count completed results, actual forwards, and identities of consumed images."""

    def __init__(self):
        self.generation = None
        self.results = deque(maxlen=512)
        self.forwards = deque(maxlen=512)
        self.result_count = self.forward_count = self.repeat_streak = 0
        self.previous_action = None
        self.sensors = {}
        self.forward_inputs = {}
        self.last_forward = None

    def update(self, *, seq, generation, observed, started, finished, action, sensors, inference):
        if generation != self.generation:
            self.__init__()
            self.generation = generation
        action = np.asarray(action, dtype=float)
        delta = None if self.previous_action is None else float(np.max(abs(action - self.previous_action)))
        exact = self.previous_action is not None and np.array_equal(action, self.previous_action)
        near = delta is not None and delta <= SAME_ACTION_ATOL
        self.repeat_streak = self.repeat_streak + 1 if near else 0
        self.previous_action = action.copy()
        self.results.append(finished)
        self.result_count += 1
        forward_calls = inference.get("forwards_this_call", 0)
        self.forward_count += forward_calls
        if forward_calls:
            self.forwards.extend([finished] * forward_calls)
            self.last_forward = finished
        rows = []
        for name, data in sensors.get("sensors", {}).items():
            identity = data.get("frame_id")
            state = self.sensors.setdefault(name, {"id": None, "events": deque(maxlen=512), "reuse_count": 0})
            reused = identity is not None and state["id"] == identity
            if identity is not None:
                if reused:
                    state["reuse_count"] += 1
                else:
                    state["events"].append(finished)
                state["id"] = identity
            if forward_calls and data.get("model_input", True):
                self.forward_inputs[name] = identity
            rows.append({**data, "name": name, "input_events": _events(state["events"], finished),
                         "frame_reused": reused, "reuse_count": state["reuse_count"],
                         "identity_known": identity is not None,
                         "used_on_last_forward": identity is not None and self.forward_inputs.get(name) == identity})
        return {"seq": seq, "generation": generation, "observed": observed, "started": started,
                "finished": finished, "input_mode": sensors.get("mode", "unknown"),
                "bundle_sequence": sensors.get("bundle_sequence"),
                "result_events": _events(self.results, finished),
                "forward_events": _events(self.forwards, finished),
                "last_forward": self.last_forward, "result_count": self.result_count,
                "forward_count": self.forward_count, "forward_known": bool(inference) and inference.get("output_source") != "unknown",
                "output_source": inference.get("output_source", "unknown"),
                "temporal_ensemble_coeff": inference.get("temporal_ensemble_coeff"),
                "n_action_steps": inference.get("n_action_steps"),
                "output_exact_same": bool(exact), "output_near_same": bool(near),
                "output_repeat_streak": self.repeat_streak, "max_action_delta_rad": delta,
                "same_action_atol_rad": SAME_ACTION_ATOL, "sensors": rows}


def diagnostic_view(worker, sender, *, now=None, phase="idle"):
    """Age telemetry on every HTTP poll, including when producers stop updating."""
    now = time.monotonic() if now is None else now
    active = phase in ("policy", "resuming", "starting", "arming")
    sender = copy.deepcopy(sender or {})
    sender["accepted_result_age_ms"] = _age(sender.get("observed"), now)
    sender["applied_age_ms"] = _age(sender.get("applied_at"), now)
    if not active:
        sender["status"] = "inactive"
    elif sender.get("holding"):
        sender["status"] = "stale_hold"
    elif sender.get("entry_until", 0) and sender["entry_until"] > now:
        sender["status"] = "first_frame_bridge"
    else:
        sender["status"] = {"rejected": "retained_trajectory", "applied": "applying",
                            "unchanged_goal": "same_goal"}.get(sender.get("replan_status"), "unknown")
    base = {"available": bool(worker), "window_s": WINDOW_S, "sender": sender,
            "result_status": "waiting" if active else "inactive", "model_status": "unknown" if active else "inactive",
            "result_hz": None, "model_forward_hz": None, "last_result_age_ms": None,
            "last_forward_age_ms": None, "result_count": 0, "forward_count": 0,
            "seq": None, "generation": None, "output_source": "unknown",
            "temporal_ensemble_coeff": None, "n_action_steps": None,
            "output_exact_same": False, "output_near_same": False, "output_repeat_streak": 0,
            "max_action_delta_rad": None, "sensors": []}
    if not worker:
        return base
    base.update({key: worker[key] for key in ("result_count", "forward_count", "seq", "generation",
                "output_source", "temporal_ensemble_coeff", "n_action_steps", "output_exact_same",
                "output_near_same", "output_repeat_streak", "max_action_delta_rad")})
    age = _age(worker["finished"], now)
    base.update(result_hz=_hz(worker["result_events"], now) if active else 0.,
                model_forward_hz=(_hz(worker["forward_events"], now) if active else 0.) if worker["forward_known"] else None,
                last_result_age_ms=age, last_forward_age_ms=_age(worker["last_forward"], now))
    if active:
        base["result_status"] = "fresh" if age is not None and age <= STALE_S * 1000 else "stale"
        base["model_status"] = worker["output_source"] if base["result_status"] == "fresh" else "stale"
    for sensor in worker["sensors"]:
        row = {k: v for k, v in sensor.items() if k not in ("input_events", "receive_events")}
        sensor_age = _age(sensor.get("received_at"), now)
        row.update(input_hz=_hz(sensor["input_events"], now) if sensor["identity_known"] else None,
                   receive_hz=_hz(sensor.get("receive_events"), now), receive_age_ms=sensor_age)
        if not active:
            row.update(status="inactive", input_hz=0.)
        elif worker["input_mode"] == "sample":
            row["status"] = "sample"
        elif not sensor["identity_known"]:
            row["status"] = "unknown"
        elif sensor_age is None or sensor_age > STALE_S * 1000:
            row["status"] = "stale"
        else:
            row["status"] = "reused" if sensor["frame_reused"] else "fresh"
        base["sensors"].append(row)
    return base
