"""Prepare a compatible policy change away from the fixed-rate command loop."""

from types import SimpleNamespace

import numpy as np

from .act_handoff import configure_loaded_arrival, handoff_target_class


MODEL_FIELDS = frozenset((
    "checkpoint", "dataset_info", "dataset_repo_id", "dataset_task", "sample",
    "pregrasp_pose", "inference_server", "output_dir", "init_pose_on_start", "config",
    "recording_policy_id", "recording_policy_name", "recording_repo_id",
))


def prepare_switch(policy_id, current_cfg, next_cfg, contract, axes, limits,
                   reference, kp, kd, reset_type, load_contract, target_factory):
    """All file reads, contract checks and polynomial bounds run in the UI thread."""
    if current_cfg.get("display_data"):
        raise ValueError("live policy changes require display_data to be disabled")
    if current_cfg.get("record_dataset") and not current_cfg.get("annotate_policy"):
        raise ValueError("live policy changes while recording require per-frame policy annotation")
    changed = sorted(k for k in set(current_cfg) | set(next_cfg)
                     if k not in MODEL_FIELDS and current_cfg.get(k) != next_cfg.get(k))
    if changed:
        raise ValueError(f"incompatible live control configuration: {changed}")
    next_contract, next_axes, _, next_limits = load_contract(next_cfg)
    for key in ("owned", "state_indices", "action_indices", "action_columns", "state_columns",
                "joint_tau_indices", "images", "fps", "policy_type", "pi05_tactile_padding"):
        if getattr(contract, key, None) != getattr(next_contract, key, None):
            raise ValueError(f"incompatible live policy contract: {key}")
    if tuple(axes) != tuple(next_axes) or not all(
        np.array_equal(a, b) for a, b in zip(limits, next_limits, strict=True)
    ):
        raise ValueError("incompatible live policy axes or joint limits")
    reference = reference.copy()
    base_target = target_factory(next_cfg)
    target_type = handoff_target_class(base_target, reference)
    target = target_type(reference, next_contract, axes, limits, dict(next_cfg))

    class PreparedReset(reset_type):
        def plan(self, now):
            prepared = getattr(self, "prepared_move", None)
            if prepared is None:
                return super().plan(now)
            curve, goal, start = prepared
            if not np.array_equal(self.q, start):
                raise RuntimeError("prepared handoff no longer matches the held command")
            self.prepared_move = None
            self.curve, self.goal = curve, goal
            self.curve.start_s = now
            self.transition("moving", now)
            self.events[-1].update(duration_s=curve.duration_s, goal=goal.tolist())

    reset = PreparedReset(target, next_cfg["pregrasp_pose"], kp, kd,
                          execute=next_cfg["output_mode"] == "dds")
    reset.plan(0.0)
    reset.prepared_move = (reset.curve, reset.goal.copy(), reference.copy())
    reset.curve = None
    reset.events.clear()
    configure_loaded_arrival(reset)
    reset.phase = "paused"
    return SimpleNamespace(policy_id=policy_id, config=dict(next_cfg), contract=next_contract,
                           target=target, reset=reset, base_target=base_target,
                           reference=reference, kp=kp.copy(), kd=kd.copy())
