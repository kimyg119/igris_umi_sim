"""Local, single-session policy service. Owns the model, never sensors or robot transport."""

import fcntl
import hashlib
import io
import json
import os
import stat
import threading
from contextlib import suppress
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

from .act_attention import attention_image_names, validate_attention_maps
from .act_policy import PolicyContract
from .act_runtime import ActInference


def model_spec(cfg):
    checkpoint = Path(cfg["checkpoint"]).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    info = json.loads(Path(cfg["dataset_info"]).read_text())
    coefficient = cfg.get("temporal_ensemble_coeff")
    if coefficient is None:
        coefficient = config.get("temporal_ensemble_coeff")
    pi05 = config.get("type") == "pi05"
    groot = config.get("type") == "groot"
    if pi05 or groot:
        if not isinstance(cfg.get("dataset_task"), str) or not cfg["dataset_task"].strip():
            raise ValueError("VLA requires the training task text in dataset_task")
        if coefficient is not None or cfg.get("head_cam_attention") or cfg.get("tactile_attention"):
            raise ValueError("ACT ensemble/attention options do not apply to VLA policies")
    elif type(coefficient) not in (int, float) or not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("persistent ACT requires a finite nonnegative temporal ensemble coefficient")
    for key in ("head_cam_attention", "tactile_attention"):
        if type(cfg.get(key, False)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    files = {}
    for path in sorted(checkpoint.rglob("*") if pi05 or groot else checkpoint.iterdir()):
        if path.is_file() and (path.suffix in (".json", ".safetensors") or (groot and path.suffix == ".txt")):
            with path.open("rb") as stream:
                files[path.relative_to(checkpoint).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    identity = {
        "checkpoint": str(checkpoint),
        "files": files,
        "dataset_info": info,
        "device": cfg.get("device", "cuda"),
        "temporal_ensemble_coeff": coefficient,
        "head_cam_attention": cfg.get("head_cam_attention", False),
        "tactile_attention": cfg.get("tactile_attention", False),
    }
    if pi05:
        identity.update(
            task=cfg["dataset_task"],
            inference_dtype="bfloat16",
            n_action_steps=1,
            compile_model=identity["device"].startswith("cuda"),
        )
    if groot:
        identity.update(task=cfg["dataset_task"], action_selection="decoded_chunk_first", chunk_size=16)
    return identity, config, info


def send_json(connection, value):
    connection.send_bytes(json.dumps(value).encode())


def validate_observation(observation, contract):
    shapes = {"observation.state": (len(contract.state_indices),)}
    if getattr(contract, "joint_tau_indices", ()):
        shapes["observation.environment_state"] = (len(contract.joint_tau_indices),)
    shapes.update({"observation.images." + k: tuple(v) for k, v in contract.images.items()})
    if set(observation) != set(shapes):
        raise ValueError("observation features differ from resident checkpoint")
    for key, shape in shapes.items():
        value = observation[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid observation shape/values: {key}")
        if key.startswith("observation.images.") and value.dtype != np.uint8:
            raise ValueError(f"expected RGB uint8 HWC: {key}")


def validate_inference_diagnostics(value):
    """Validate optional forward evidence at the resident-process boundary."""
    if value is None or value == {}:
        return {}
    fields = {
        "forward_count", "forwards_this_call", "forward_completed_at", "forward_duration_s",
        "output_source", "temporal_ensemble_coeff", "n_action_steps", "chunk_size",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid inference diagnostics fields")
    for key in ("forward_count", "forwards_this_call"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"invalid inference diagnostics {key}")
    for key in ("forward_completed_at", "forward_duration_s", "temporal_ensemble_coeff"):
        number = value[key]
        if number is not None and (type(number) not in (int, float) or not np.isfinite(number) or number < 0):
            raise ValueError(f"invalid inference diagnostics {key}")
    for key in ("n_action_steps", "chunk_size"):
        if value[key] is not None and (type(value[key]) is not int or value[key] <= 0):
            raise ValueError(f"invalid inference diagnostics {key}")
    source = value["output_source"]
    if source not in ("fresh_forward", "cached_action_chunk", "unknown"):
        raise ValueError("invalid inference diagnostics output_source")
    count, calls = value["forward_count"], value["forwards_this_call"]
    completed, duration = value["forward_completed_at"], value["forward_duration_s"]
    if (calls > count
            or (count == 0 and (completed is not None or duration is not None))
            or (count > 0 and (completed is None or duration is None))
            or (source == "fresh_forward" and calls == 0)
            or (source == "cached_action_chunk" and (calls != 0 or count == 0))
            or (source == "unknown" and count != 0)):
        raise ValueError("inconsistent inference diagnostics evidence")
    return dict(value)


def handle_client(connection, inference, identity, session_lock):
    acquired = False
    try:
        if not connection.poll(10):
            return
        request = json.loads(connection.recv_bytes(1 << 20))
        if request.get("op") != "open" or request.get("identity") != identity:
            raise ValueError("resident model/config differs; restart ACT server with the matching YAML")
        acquired = session_lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("ACT server busy: another inference session is active")
        inference.reset()
        send_json(connection, {"ok": True, "pid": os.getpid()})
        while connection.poll(30):
            request = json.loads(connection.recv_bytes(4096))
            if request.get("op") == "reset":
                inference.reset()
                send_json(connection, {"ok": True})
            elif request.get("op") == "infer":
                if not connection.poll(10):
                    raise TimeoutError("observation payload timed out")
                # NPZ with allow_pickle=False keeps IPC free of executable object deserialization.
                with np.load(io.BytesIO(connection.recv_bytes(32 << 20)), allow_pickle=False) as archive:
                    observation = dict(archive)
                validate_observation(observation, inference.contract)
                action = inference(observation)
                reply = {"ok": True, "action": action.tolist()}
                diagnostics = validate_inference_diagnostics(getattr(inference, "last_diagnostics", None))
                if diagnostics:
                    reply["inference_diagnostics"] = diagnostics
                if identity.get("head_cam_attention") or identity.get("tactile_attention"):
                    reply["image_attention"] = {k: v.tolist() for k, v in inference.attention_maps.items()}
                send_json(connection, reply)
            else:
                raise ValueError("unknown ACT server operation")
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except Exception as exc:
        with suppress(OSError):
            send_json(connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if acquired:
            try:
                inference.reset()
            finally:
                session_lock.release()
        connection.close()


class RemoteActInference:
    def __init__(self, cfg):
        identity, config, info = model_spec(cfg)
        self.identity = identity
        self.contract = PolicyContract(config, info)
        self.attention_names = attention_image_names(
            self.contract.images,
            head_cam_attention=cfg.get("head_cam_attention", False),
            tactile_attention=cfg.get("tactile_attention", False),
        )
        self.attention_maps = {}
        self.last_diagnostics = {}
        self.connection = Client(cfg["inference_server"], family="AF_UNIX")
        try:
            send_json(self.connection, {"op": "open", "identity": identity})
            self.server_pid = self._reply()["pid"]
        except BaseException:
            self.close()
            raise

    def _reply(self):
        if not self.connection.poll(10):
            raise TimeoutError("resident ACT server did not respond within 10 seconds")
        reply = json.loads(self.connection.recv_bytes(1 << 20))
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply

    def reset(self):
        self.last_diagnostics = {}
        send_json(self.connection, {"op": "reset"})
        self._reply()
        self.attention_maps = {}

    def __call__(self, observation):
        self.attention_maps = {}
        self.last_diagnostics = {}
        validate_observation(observation, self.contract)
        with io.BytesIO() as buffer:
            np.savez(buffer, **observation)
            send_json(self.connection, {"op": "infer"})
            self.connection.send_bytes(buffer.getvalue())
        reply = self._reply()
        action = np.asarray(reply["action"], dtype=float)
        if action.shape != (len(self.contract.owned),) or not np.isfinite(action).all():
            raise ValueError("invalid resident ACT action")
        if self.attention_names:
            self.attention_maps = validate_attention_maps(reply.get("image_attention"), self.attention_names)
        self.last_diagnostics = validate_inference_diagnostics(reply.get("inference_diagnostics"))
        return action

    def close(self):
        self.connection.close()


def serve(cfg):
    import torch

    address = Path(cfg["inference_server"])
    address.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = address.parent.stat()
    if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) != 0o700:
        raise PermissionError("ACT socket must be in a private, owner-only (0700) directory")
    with address.with_suffix(".lock").open("a") as lockfile:
        # The file lock survives a stale socket and prevents replacing an active server.
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if address.exists():
            if not stat.S_ISSOCK(address.stat().st_mode):
                raise ValueError("ACT socket path contains a non-socket file")
            address.unlink()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        identity, _, info = model_spec(cfg)
        inference = ActInference(
            Path(identity["checkpoint"]),
            info,
            identity["device"],
            temporal_ensemble_coeff=identity["temporal_ensemble_coeff"],
            head_cam_attention=identity["head_cam_attention"],
            tactile_attention=identity["tactile_attention"],
            task=identity.get("task"),
        )
        observation = {"observation.state": np.zeros(len(inference.contract.state_indices), np.float32)}
        if inference.contract.joint_tau_indices:
            observation["observation.environment_state"] = np.zeros(
                len(inference.contract.joint_tau_indices), np.float32
            )
        observation.update(
            {
                "observation.images." + key: np.zeros(shape, np.uint8)
                for key, shape in inference.contract.images.items()
            }
        )
        for _ in range(8):
            inference(observation)
        inference.reset()
        # Pi0.5 compilation can take minutes; expose the socket only after warmup succeeds.
        with Listener(str(address), family="AF_UNIX") as listener:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "pid": os.getpid(),
                        "socket": str(address),
                        "device": identity["device"],
                        "model_sha256": identity["files"]["model.safetensors"],
                        "hardware_executed": False,
                    }
                ),
                flush=True,
            )
            session_lock = threading.Lock()
            while True:
                connection = listener.accept()
                threading.Thread(
                    target=handle_client, args=(connection, inference, identity, session_lock), daemon=True
                ).start()
