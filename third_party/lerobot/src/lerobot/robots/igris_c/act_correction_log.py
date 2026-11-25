"""Bounded, read-only web history of accepted boundary-corrected targets."""

from collections import deque

from .igris_c import joint_names


_JOINT_NAMES = joint_names()


def build_correction_snapshot(target, command, *, at, policy_id, generation):
    """Capture after a command write; a corrected goal need not be the active trajectory."""
    planned = target.planned_endpoint
    status = target.replan_status
    if (target.pending is not None or getattr(target, "entry_deferred", None) is not None
            or (status == "applied" and target.applied_seq != target.seq)):
        status = "waiting"
    return {
        "policy_id": policy_id,
        "generation": generation,
        "policy_seq": target.seq,
        "at": at,
        "projected": target.target_projected,
        "holding": target.holding,
        "replan_status": status,
        "applied_seq": target.applied_seq,
        "max_correction_rad": target.max_correction_rad,
        "changed_axes": list(target.correction_axes),
        "correction_warnings": [dict(warning) for warning in target.correction_warnings],
        "rows": [
            {"axis": axis, "name": _JOINT_NAMES[axis],
             "requested": float(target.requested_goal[axis]),
             "corrected": float(target.goal[axis]),
             "delta": float(target.goal[axis] - target.requested_goal[axis]),
             "trajectory_target": float(planned[0][axis]) if planned is not None else None,
             "command": float(command[axis])}
            for axis in target.axes
        ] if target.seq >= 0 and target.target_projected else [],
    }


class CorrectionLog:
    """Called under the web lock; published events are never mutated afterward."""

    def __init__(self):
        self.events = deque(maxlen=30)
        self.clear()

    def clear(self):
        self.events.clear()
        self.latest = None
        self.total_events = 0
        self.current = None
        self.last_identity = None
        self.started_at = None

    def record(self, payload):
        if payload["policy_seq"] < 0:
            return
        if self.started_at is None:
            self.started_at = payload["at"]
        self.current = {key: payload[key] for key in (
            "policy_seq", "generation", "projected", "holding", "replan_status", "applied_seq",
            "correction_warnings",
        )}
        identity = (payload["policy_id"], payload["generation"], payload["policy_seq"])
        if identity == self.last_identity:
            return
        self.last_identity = identity
        if not payload["projected"] or payload["holding"]:
            return
        self.total_events += 1
        event = {**payload, "id": self.total_events,
                 "elapsed_s": max(0., payload["at"] - self.started_at)}
        self.latest = event
        self.events.append({**event, "rows": [row for row in event["rows"]
                                              if row["axis"] in event["changed_axes"]]})

    def snapshot(self):
        return {"capacity": self.events.maxlen, "total_events": self.total_events,
                "latest": self.latest,
                "events": list(reversed(self.events)), "current": self.current}
