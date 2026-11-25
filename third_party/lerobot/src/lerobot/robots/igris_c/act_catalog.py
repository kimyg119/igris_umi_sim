"""Explicit policy catalog and ownership-aware reuse of resident ACT servers."""

import copy
import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .act_server import RemoteActInference

MODEL_KEYS = (
    "checkpoint",
    "dataset_info",
    "sample",
    "inference_server",
    "temporal_ensemble_coeff",
    "cameras",
    "tactile",
    "ros2_bundle",
    "dataset_repo_id",
    "dataset_task",
    "pregrasp_pose",
    "waist_yaw_step_gate",  # Explicit per-policy exception to the shared target-step guard.
)


def load_catalog(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"control_config", "policies"}:
        raise ValueError("catalog requires control_config and policies")
    base = yaml.safe_load(Path(data["control_config"]).read_text())
    entries = data["policies"]
    if not isinstance(base, dict) or not isinstance(entries, dict) or not entries:
        raise ValueError("catalog requires a control mapping and nonempty policy mapping")
    policies = {}
    for key, entry in entries.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", key):
            raise ValueError("policy IDs must be short lowercase names")
        if not isinstance(entry, dict) or set(entry) != {"name", "config"}:
            raise ValueError("each policy requires name and config")
        if not isinstance(entry["name"], str) or not entry["name"].strip():
            raise ValueError("policy name must be nonempty text")
        model = yaml.safe_load(Path(entry["config"]).read_text())
        if not isinstance(model, dict):
            raise ValueError("policy config must be a mapping")
        cfg = copy.deepcopy(base)
        cfg.update({k: copy.deepcopy(model[k]) for k in MODEL_KEYS if k in model})
        # The new dashboard shows observations itself; Rerun remains an explicit CLI option.
        cfg.update(display_data=False, head_cam_attention=False, tactile_attention=False)
        policies[key] = {"name": entry["name"], "config": cfg}
    return policies


class ResidentModels:
    def __init__(self, policies, output):
        self.policies = policies
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.owned = {}
        self.status = {}

    def ensure(self, key):
        process = self.owned.get(key)
        exit_code = process.poll() if process is not None else None
        previous = self.status.get(key, {})
        if previous.get("status") == "ready" and exit_code is None:
            try:
                # Health checks must not race Start for the exclusive inference session.
                os.kill(previous["pid"], 0)
                return previous
            except ProcessLookupError:
                pass
        cfg = self.policies[key]["config"]
        if process is not None and exit_code is None and not Path(cfg["inference_server"]).exists():
            # The server binds only after loading/compilation; avoid rehashing weights on every poll.
            self.status[key] = {"status": "loading", "pid": None, "error": None}
            return self.status[key]
        if exit_code is not None:
            log = (self.output / f"{key}.log").read_text(errors="replace")[-1500:]
            state = {"status": "error", "error": f"GPU server exited ({process.returncode}): {log}"}
        else:
            try:
                inference = RemoteActInference(cfg)
                try:
                    state = {"status": "ready", "pid": inference.server_pid, "error": None}
                finally:
                    inference.close()
            except RuntimeError as exc:
                # Identity is validated before the busy response; never reset/replace its owner.
                state = {"status": "busy" if "ACT server busy:" in str(exc) else "error", "error": str(exc)}
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                    state = {"status": "error", "error": str(exc)}
                else:
                    if process is None:
                        config_path = self.output / f"{key}.json"
                        config_path.write_text(json.dumps(cfg, default=str, indent=2) + "\n")
                        with (self.output / f"{key}.log").open("ab") as log:
                            self.owned[key] = subprocess.Popen(
                                [sys.executable, "examples/igris/serve_act.py", "--config", str(config_path)],
                                stdin=subprocess.DEVNULL,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                    state = {"status": "loading", "error": None}
            except Exception as exc:
                state = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        state.setdefault("pid", None)
        self.status[key] = state
        return state

    def close(self):
        errors = []
        for key in list(self.owned):
            try:
                self.unload(key)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Failed to unload GPU servers", errors)

    def unload(self, key):
        process = self.owned.get(key)
        if process is None and self.status.get(key, {}).get("status") in ("ready", "busy"):
            raise RuntimeError("cannot unload a server not owned by this web session")
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            del self.owned[key]
            Path(self.policies[key]["config"]["inference_server"]).unlink(missing_ok=True)
        self.status[key] = {"status": "unloaded", "pid": None, "error": None}
        return self.status[key]
