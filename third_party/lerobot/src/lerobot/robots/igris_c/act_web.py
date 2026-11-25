"""Web controls feeding the existing ACT sender's command queue."""

import copy
import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .act_correction_log import CorrectionLog
from .act_diagnostics import DiagnosticChannel, diagnostic_view


def preview_image_steps(checkpoint):
    steps = json.loads((Path(checkpoint) / "policy_preprocessor.json").read_text())["steps"]
    configs = [step["config"] for step in steps if step["registry_name"] == "image_crop_resize_processor"]
    if not configs:
        return []
    from lerobot.processor import ImageCropResizeProcessorStep

    return [ImageCropResizeProcessorStep(**config) for config in configs]


def preview_image_shapes(images, image_steps=()):
    shapes = dict(images)
    for step in image_steps:
        for name, (height, width, channels) in shapes.items():
            key = f"observation.images.{name}"
            if step.image_keys is not None and key not in step.image_keys:
                continue
            if step.crop_params_dict and key in step.crop_params_dict:
                height, width = step.crop_params_dict[key][2:]
            if step.resize_size is not None:
                height, width = step.resize_size
            shapes[name] = (height, width, channels)
    for side in ("left", "right"):
        if not any("tactile" in name and name.endswith(f"_{side}_heatmap") for name in images):
            shapes[f"tactile_{side}_heatmap"] = (80, 320, 3)
    return shapes


class ObservationPreview:
    """One bounded RGB snapshot; lock contention drops display frames, never actions."""

    def __init__(self, ctx, images, *, tactile_monitor=False):
        self.diagnostics = DiagnosticChannel(ctx)
        self.model_images = set(images)
        self.images = preview_image_shapes(images) if tactile_monitor else dict(images)
        self.buffer = ctx.RawArray("B", 32 + sum(int(np.prod(s)) for s in self.images.values()))
        self.available = ctx.RawArray("B", len(self.images))
        self.lock = ctx.Lock()
        self.changed = ctx.Event()
        np.frombuffer(self.buffer, np.float64, count=4)[:] = (-1, 0, 0, 0)

    def publish(self, observation, seq, observed, finished, *, started=None, diagnostics=None):
        if diagnostics is not None:
            self.diagnostics.publish(diagnostics)
        if not self.lock.acquire(False):
            return False
        try:
            offset = 32
            available = getattr(self, "available", None)
            for index, (name, shape) in enumerate(self.images.items()):
                frame = observation.get("observation.images." + name)
                if frame is None and name not in getattr(self, "model_images", self.images):
                    available[index] = 0
                    offset += int(np.prod(shape))
                    continue
                if frame is None:
                    raise ValueError(f"missing model preview image: {name}")
                if frame.shape != tuple(shape) or frame.dtype != np.uint8:
                    raise ValueError(f"invalid preview image: {name}")
                pixels = np.ndarray(shape, np.uint8, buffer=self.buffer, offset=offset)
                pixels[:] = frame
                if available is not None:
                    available[index] = 1
                offset += pixels.size
            np.frombuffer(self.buffer, np.float64, count=4)[:] = (
                seq,
                observed,
                finished,
                observed if started is None else started,
            )
        finally:
            self.lock.release()
        # A dashboard started before this update can still spawn a fresh worker.
        if changed := getattr(self, "changed", None):
            changed.set()
        return True

    def read(self):
        # A terminated producer can leave its lock held: the web must still respond.
        if not self.lock.acquire(timeout=0.05):
            return None
        try:
            seq, observed, finished, started = np.frombuffer(self.buffer, np.float64, count=4).tolist()
            if seq < 0:
                return None
            images, offset = {}, 32
            available = getattr(self, "available", None)
            for index, (name, shape) in enumerate(self.images.items()):
                images[name] = (
                    np.ndarray(shape, np.uint8, buffer=self.buffer, offset=offset).copy()
                    if available is None or available[index]
                    else None
                )
                offset += int(np.prod(shape))
            return {
                "seq": int(seq),
                "observed": observed,
                "finished": finished,
                "started": started,
                "images": images,
            }
        finally:
            self.lock.release()


class ActWebControl:
    def __init__(self, cfg, output, *, policies=None):
        self.lock = threading.RLock()
        self.corrections = CorrectionLog()
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.shutdown_requested = threading.Event()
        self.model_requests = queue.SimpleQueue()
        self.keys = None
        self.pending = None
        self.token = secrets.token_urlsafe(32)
        self.policies = copy.deepcopy(policies or {})
        for policy in self.policies.values():
            policy.setdefault("preview_images", preview_image_shapes(policy["images"]))
        self.preview_enabled = True
        self.preview = None
        self.preview_stop = None
        self.preview_thread = None
        self.preview_epoch = 0
        self.preview_images = {}
        self.observation = None
        self.state = {
            "phase": "loading",
            "checkpoint": str(Path(cfg["checkpoint"]).resolve()),
            "device": cfg["device"],
            "input_mode": cfg["input_mode"],
            "output_mode": cfg["output_mode"],
            "output_dir": str(output.resolve()),
            "pose_file": cfg["pregrasp_pose"],
            "pose_name": json.loads(Path(cfg["pregrasp_pose"]).read_text()).get("name", "저장 자세"),
            "dds_writes": 0,
            "hardware_executed": False,
            "error": None,
            "max_action_step": cfg.get("max_action_step"),
            "velocity_limit_gate": cfg.get("velocity_limit_gate", True),
            "acceleration_limit_gate": cfg.get("acceleration_limit_gate", True),
            "jerk_limit_gate": cfg.get("jerk_limit_gate", True),
            "waist_yaw_step_gate": cfg.get("waist_yaw_step_gate", True),
            "policy_id": next((key for key, p in self.policies.items() if p["status"] == "ready"), None),
        }
        if self.policies and self.state["policy_id"] is None:
            self.state["checkpoint"] = ""
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, data, content_type="application/json"):
                body = (
                    data
                    if isinstance(data, bytes)
                    else (data.encode() if isinstance(data, str) else json.dumps(data).encode())
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def same_origin_request(self):
                port = self.server.server_port
                # Accept the interface actually reached, without trusting arbitrary DNS Host names.
                hosts = {
                    f"127.0.0.1:{port}",
                    f"localhost:{port}",
                    f"{self.connection.getsockname()[0]}:{port}",
                }
                host = self.headers.get("Host")
                return host in hosts and self.headers.get("Origin") in (
                    None,
                    f"http://{host}",
                )

            def do_GET(self):  # noqa: N802 -- HTTP handler API
                if not self.same_origin_request():
                    self.reply(403, {"error": "same origin required"})
                elif self.path == "/":
                    self.reply(200, PAGE.replace("__TOKEN__", control.token), "text/html")
                elif self.path == "/status":
                    self.reply(200, control.snapshot())
                elif self.path.startswith("/observation/"):
                    with control.lock:
                        body = control.preview_images.get(self.path.partition("?")[0])
                    if body is None:
                        self.reply(404, {"error": "observation is no longer available"})
                    else:
                        self.reply(200, body, "image/jpeg")
                else:
                    self.reply(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802 -- HTTP handler API
                if not self.same_origin_request() or not secrets.compare_digest(
                    self.headers.get("X-ACT-Token", "").encode(), control.token.encode()
                ):
                    self.reply(403, {"error": "same-origin page token required"})
                    return
                if self.path not in ("/command", "/policy", "/model"):
                    self.reply(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("invalid command length")
                    self.connection.settimeout(2)
                    data = json.loads(self.rfile.read(length))
                    keys = {
                        "/command": {"action"},
                        "/policy": {"policy_id"},
                        "/model": {"action", "policy_id"},
                    }[self.path]
                    if not isinstance(data, dict) or set(data) != keys:
                        raise ValueError(f"expected {sorted(keys)}")
                    if self.path == "/command":
                        control.command(data["action"])
                    elif self.path == "/model":
                        control.manage_model(data["action"], data["policy_id"])
                    else:
                        control.select_policy(data["policy_id"])
                except (ValueError, TypeError, OSError) as exc:
                    self.reply(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self.reply(409, {"error": str(exc)})
                else:
                    self.reply(200, control.snapshot())

        host = cfg.get("web_host", "127.0.0.1")
        self.server = ThreadingHTTPServer((host, cfg["web_port"]), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        print(f"ACT web: {self.url} (bind={host}, waiting for browser Start)", flush=True)

    def update(self, phase, *, correction=None, **values):
        with self.lock:
            if self.shutdown_requested.is_set():
                phase = "shutting_down"
            if phase != self.state["phase"]:
                self.pending = None
            if correction is not None:
                self.corrections.record(correction)
            self.state.update(phase=phase, **values)

    def actions(self):
        if self.shutdown_requested.is_set():
            return []
        phase = self.state["phase"]
        extra = ["shutdown"] if self.policies else []
        if getattr(self, "policies", None) and phase in ("idle", "finished", "error"):
            policy = self.policies.get(self.state["policy_id"], {})
            changing = any(p["status"] in ("loading", "unloading") for p in self.policies.values())
            return (
                ["start"]
                if policy.get("status") == "ready" and not self.started.is_set() and not changing
                else []
            ) + extra
        if self.stopped.is_set() or phase in ("finished", "error"):
            return extra
        action = {"idle": "start", "policy": "pause", "waiting": "resume"}.get(phase)
        return ([action] if action and self.pending is None else []) + ["stop"] + extra

    def snapshot(self):
        with self.lock:
            preview = self.preview
            corrections = self.corrections.snapshot()
            sender_diagnostics = copy.deepcopy(self.state.get("sender_diagnostics", {}))
            observation = copy.deepcopy(self.observation)
            if observation is not None:
                observation["age_ms"] = max(0.0, (time.monotonic() - observation["observed"]) * 1000)
                observation["stale"] = observation["age_ms"] > 500 or self.state["phase"] != "policy"
            result = {
                **self.state,
                "actions": self.actions(),
                "pending": self.pending,
                "policies": [
                    {"id": key, **copy.deepcopy(p), "model_actions": self.model_actions(p)}
                    for key, p in self.policies.items()
                ],
                "can_select": self._can_select(),
                "observation": observation,
            }
        result["correction_log"] = copy.deepcopy(corrections)
        result["diagnostics"] = diagnostic_view(
            preview.diagnostics.read() if preview is not None else None,
            sender_diagnostics,
            now=time.monotonic(),
            phase=result["phase"],
        )
        return result

    def _can_select(self):
        return (
            bool(self.policies)
            and not self.shutdown_requested.is_set()
            and not self.started.is_set()
            and self.state["phase"] in ("idle", "finished", "error")
        )

    def _clear_preview(self):
        self.preview_epoch += 1
        self.preview = None
        self.preview_images = {}
        self.observation = None
        self.state.pop("sender_diagnostics", None)
        self.corrections.clear()

    def select_policy(self, policy_id):
        with self.lock:
            if not isinstance(policy_id, str) or policy_id not in self.policies:
                raise ValueError("unknown policy")
            if not self._can_select():
                raise RuntimeError("실행을 종료한 후 정책을 변경할 수 있습니다.")
            if self.policies[policy_id]["status"] != "ready":
                raise RuntimeError("GPU 로딩이 완료된 정책만 선택할 수 있습니다.")
            self._select_policy(policy_id)

    def _select_policy(self, policy_id):
        self.state.update(
            policy_id=policy_id,
            checkpoint=self.policies[policy_id]["checkpoint"] if policy_id else "",
            pose_file=self.policies[policy_id]["pose_file"] if policy_id else None,
            pose_name=self.policies[policy_id]["pose_name"] if policy_id else "정책 선택 대기",
            waist_yaw_step_gate=self.policies[policy_id].get("waist_yaw_step_gate", True) if policy_id else True,
            server_pid=self.policies[policy_id].get("pid") if policy_id else None,
            phase="idle",
            error=None,
            command_count=0,
            policy_seq=None,
            dds_writes=0,
            hardware_executed=False,
        )
        self._clear_preview()

    def model_actions(self, policy):
        if not self._can_select():
            return []
        return {"unloaded": ["load"], "ready": ["unload"], "error": ["load", "unload"]}.get(
            policy["status"], []
        )

    def manage_model(self, action, policy_id):
        with self.lock:
            if not isinstance(action, str) or action not in ("load", "unload"):
                raise ValueError("expected load or unload")
            if not isinstance(policy_id, str) or policy_id not in self.policies:
                raise ValueError("unknown policy")
            policy = self.policies[policy_id]
            if action not in self.model_actions(policy):
                raise RuntimeError("추론을 종료한 뒤 GPU 모델을 관리할 수 있습니다.")
            policy.update(status="loading" if action == "load" else "unloading", error=None)
            self.model_requests.put((action, policy_id))

    def set_resident(self, policy_id, status):
        with self.lock:
            if self.shutdown_requested.is_set():
                return
            if self.policies[policy_id]["status"] == "unloading" and status["status"] not in (
                "unloaded",
                "error",
            ):
                return  # A health check begun before Unload must not re-enable Start.
            self.policies[policy_id].update(status)
            if status["status"] == "unloaded" and self.state["policy_id"] == policy_id:
                self._select_policy(None)
            elif status["status"] == "ready" and self.state["policy_id"] is None:
                self._select_policy(policy_id)

    def finish_run(self, result):
        with self.lock:
            if self.policies and "ACT server busy:" in (result.get("error") or ""):
                self.policies[self.state["policy_id"]].update(status="busy", error=result["error"])
            self.state.update(
                phase="shutting_down"
                if self.shutdown_requested.is_set()
                else "error"
                if result.get("error")
                else "finished",
                error=result.get("error"),
                dds_writes=result.get("dds_writes", 0),
                hardware_executed=result.get("hardware_executed", False),
            )
            self.started.clear()
            self.stopped.clear()
            self.pending = None
            self.keys = None

    def start_preview(self, ctx, images, *, checkpoint=None):
        self.stop_preview()
        preview = ObservationPreview(ctx, images, tactile_monitor=True)
        image_steps = preview_image_steps(checkpoint) if checkpoint is not None else []
        display_shapes = preview_image_shapes(images, image_steps)
        with self.lock:
            self._clear_preview()
            self.preview = preview
            epoch = self.preview_epoch
        stopped = self.preview_stop = threading.Event()

        def consume():
            previous = -1
            while not stopped.is_set():
                if not preview.changed.wait(0.1):
                    continue
                preview.changed.clear()
                if stopped.is_set():
                    break
                frame = preview.read()
                if frame is None or frame["seq"] == previous:
                    continue
                encoded, metadata = {}, []
                try:
                    if image_steps:
                        import torch

                        from lerobot.policies.utils import prepare_observation_for_inference

                        # Apply saved spatial preprocessing only in the display thread, before normalization.
                        with torch.inference_mode():
                            batch = prepare_observation_for_inference(
                                {f"observation.images.{name}": frame["images"][name] for name in images},
                                torch.device("cpu"),
                            )
                            for step in image_steps:
                                batch = step.observation(batch)
                            for name in images:
                                pixels = batch[f"observation.images.{name}"][0].permute(1, 2, 0)
                                frame["images"][name] = pixels.mul(255).round().clamp(0, 255).byte().numpy()
                    for name, rgb in frame["images"].items():
                        url = None
                        if rgb is not None:
                            ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if not ok:
                                raise ValueError(f"preview JPEG failed: {name}")
                            url = f"/observation/{epoch}/{frame['seq']}/{name}"
                            encoded[url] = jpeg.tobytes()
                        shape = display_shapes[name]
                        metadata.append(
                            {
                                "name": name,
                                "height": shape[0],
                                "width": shape[1],
                                "url": url,
                                "available": rgb is not None,
                                "model_input": name in images,
                            }
                        )
                except Exception as exc:
                    with self.lock:
                        self.state["preview_error"] = str(exc)
                    continue
                with self.lock:
                    if epoch != self.preview_epoch:
                        return
                    self.preview_images.update(encoded)
                    # Keep two complete observations so delayed HTTP reads never receive newer pixels.
                    self.preview_images = dict(list(self.preview_images.items())[-2 * len(preview.images) :])
                    self.observation = {
                        "seq": frame["seq"],
                        "observed": frame["observed"],
                        "result_latency_ms": (frame["finished"] - frame["observed"]) * 1000,
                        "inference_ms": (frame["finished"] - frame["started"]) * 1000,
                        "images": metadata,
                    }
                previous = frame["seq"]

        self.preview_thread = threading.Thread(target=consume, daemon=True, name="act-web-preview")
        self.preview_thread.start()
        return preview

    def stop_preview(self):
        if self.preview_stop is not None:
            self.preview_stop.set()
            self.preview_thread.join(timeout=2)
            self.preview_stop = self.preview_thread = None

    def command(self, action):
        if not isinstance(action, str):
            raise ValueError("action must be text")
        with self.lock:
            if action not in self.actions():
                raise RuntimeError("현재 상태에서 실행할 수 없는 명령입니다.")
            if action == "shutdown":
                self.shutdown_requested.set()
                self.stopped.set()
                self.state["phase"] = "shutting_down"
            elif action == "stop":
                self.stopped.set()
            elif action == "start":
                if getattr(self, "policies", None):
                    self.stopped.clear()
                    self._clear_preview()
                    self.state.update(
                        error=None,
                        preview_error=None,
                        dds_writes=0,
                        hardware_executed=False,
                        command_count=0,
                        policy_seq=None,
                    )
                self.state["phase"] = "starting"
                self.started.set()
            else:
                self.keys.put(("p" if action == "pause" else "space", time.monotonic()))
            self.pending = action

    def close(self):
        self.stop_preview()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IGRIS · Policy Studio</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#0b1018;color:#e7edf8}
*{box-sizing:border-box}body{max-width:1460px;margin:auto;padding:30px 32px 40px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px;gap:20px}
.brand{letter-spacing:2px;color:#83b7ff;font-size:12px;font-weight:700}h1{font-size:30px;margin:8px 0}
h2{font-size:17px;margin:0 0 16px}.sub,dt{color:#91a1b8;font-size:13px;line-height:1.6}
.layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:20px}.stack{display:grid;gap:20px;align-content:start}
.card{background:#131c29;border:1px solid #27344a;border-radius:16px;padding:22px;min-width:0}
.top{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}
.badge{font-size:12px;padding:7px 11px;border-radius:20px;background:#20324b;color:#b5d3fc}
#phase{font-size:23px;font-weight:650}.dot{width:8px;height:8px;border-radius:50%;background:#58d4ad;display:inline-block;margin-right:8px}
label{display:block;font-size:13px;margin-bottom:9px;color:#b8c7dc}
select{width:100%;font:inherit;border:1px solid #405574;background:#0e1724;color:#edf5ff;padding:13px 10px;border-radius:9px}
select:disabled{opacity:.55}.model-buttons{display:flex;gap:8px;margin-top:10px}.model-buttons button{padding:8px 10px;font-size:12px}.model-error{color:#ffb8b8;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}#shutdown{background:#552b35;border-color:#ad5765}.model{padding:15px 0;border-bottom:1px solid #253247}.model:last-child{border-bottom:0;padding-bottom:0}
.model-name{font-size:13px;font-weight:600;line-height:1.7}.model-state{font-size:12px;color:#7fe0bb;margin-top:5px}
.model-state.bad{color:#f3ba81}.buttons{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
button{font:inherit;font-size:13px;padding:13px 16px;border:1px solid #3d526f;border-radius:9px;background:#21334e;color:#eff6ff;cursor:pointer}
button[data-action=start]{background:#2869d4;border-color:#4084ef}button[data-action=stop]{background:#44232d;border-color:#6e3442}
button:disabled{opacity:.35;cursor:not-allowed}button:focus-visible,select:focus-visible{outline:3px solid #98cfff;outline-offset:3px}
#hint{margin:12px 0 0}#error{color:#ffb8b8;white-space:pre-wrap;font-size:13px;margin:12px 0 0}
.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:18px}
figure{margin:0;background:#0b121e;border:1px solid #29384d;border-radius:11px;overflow:hidden}
.image-area{min-height:170px;aspect-ratio:4/3;display:grid;place-items:center;position:relative;background:#080d15}
figure img{width:100%;height:100%;position:absolute;object-fit:contain}figure .empty{font-size:13px;color:#697c97;text-align:center;padding:15px}
figcaption{padding:12px;font-size:12px;color:#b7c6db;display:flex;justify-content:space-between;gap:10px}
#feed-state.stale{color:#edbd7c;background:#352b20}#feed-state{color:#7fe0bb;background:#153a33}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:18px 0 0}.metric{border-left:2px solid #324963;padding-left:12px}
.metric strong{display:block;font-size:20px;font-weight:600;margin-bottom:3px}.metric span{font-size:11px;color:#91a1b8}
dl{display:grid;grid-template-columns:105px minmax(0,1fr);gap:10px;font-size:12px;margin:0}dd{margin:0;overflow-wrap:anywhere;line-height:1.7}
.note{border-top:1px solid #27344a;padding-top:14px;margin:18px 0 0}.limit{font-size:24px;font-weight:600;margin:0}.limit small{font-size:13px;color:#91a1b8;font-weight:400}
@media(max-width:900px){.layout{grid-template-columns:1fr}.images{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}body{padding:22px 16px}.stack{gap:14px}}
@media(max-width:500px){.images{grid-template-columns:1fr}button{flex:1 1 45%}.metrics{gap:8px}.metric strong{font-size:17px}}

.diagnostic-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:16px;margin:18px 0}
.diagnostic-badges{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.diag-pill{font-size:12px;padding:7px 10px;border-radius:9px;background:#20324b;color:#b5d3fc}
.diag-pill.good{background:#153a33;color:#7fe0bb}.diag-pill.warn{background:#352b20;color:#edbd7c}.diag-pill.bad{background:#44232d;color:#ffb8b8}
.diag-pill.muted{background:#202836;color:#a1aec1}.diagnostic-detail{font-size:12px;color:#b8c7dc;line-height:1.7;overflow-wrap:anywhere;margin:10px 0 0}
#sender-reason{padding:12px;border:1px solid #785e35;background:#2b251d;border-radius:9px;color:#f3d7a4;white-space:pre-wrap}
.sensor-table-wrap{overflow:auto;margin-top:16px}.sensor-table{width:100%;border-collapse:collapse;min-width:650px;font-size:12px}
.sensor-table th{text-align:left;color:#91a1b8;font-weight:500;padding:10px 8px;border-bottom:1px solid #40506a;white-space:nowrap}
.sensor-table td{padding:10px 8px;border-bottom:1px solid #27344a;vertical-align:top;line-height:1.5}.sensor-table td.frame-id{max-width:180px;overflow-wrap:anywhere}
.correction-table-wrap{max-height:320px;overflow:auto;margin-top:14px}.correction-table{min-width:860px;font-variant-numeric:tabular-nums}
.correction-table th{position:sticky;top:0;background:#131c29;z-index:1}.correction-table td:not(:first-child){white-space:nowrap;text-align:right;font-family:ui-monospace,monospace}
.correction-table .changed{background:#352b20;color:#ffe0a9}.correction-table .unchanged{color:#b8c7dc}.correction-table td:first-child{min-width:215px}
.correction-history-details{border-top:1px solid #27344a;margin-top:18px;padding-top:14px}.correction-history-details summary{cursor:pointer;color:#b8c7dc;font-size:13px}
.correction-history{padding-left:22px;margin:12px 0 0;max-height:240px;overflow:auto;font-size:12px;line-height:1.7}.correction-history li{padding:8px 0;border-bottom:1px solid #27344a}
.correction-history .correction-log-values{color:#edbd7c;white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,monospace}
#correction-summary{white-space:pre-wrap;color:#ffe0a9;padding:10px 12px;border-left:2px solid #d3a457;background:#211e19;border-radius:6px;font-family:ui-monospace,monospace}
.correction-warning{white-space:pre-wrap;color:#ffe0a9;background:#302619;border-left:3px solid #e3ab43;padding:10px 12px;border-radius:6px;overflow-wrap:anywhere}
.correction-log-warning{white-space:pre-wrap;color:#ffe0a9}
</style></head><body>
<header><div><div class="brand">IGRIS / LEROBOT</div><h1>Policy Studio</h1><div class="sub">정책을 선택하고, 모델이 보는 입력을 함께 확인하세요.</div></div><div class="top"><span id="mode" class="badge">연결 중</span><button id="shutdown" data-action="shutdown" disabled>웹 종료 · GPU 해제</button></div></header>
<div class="layout"><aside class="stack">
<section class="card" id="selector-card"><h2>GPU 모델</h2><p class="sub">필요한 체크포인트만 GPU에 올리세요. 모델 로딩만으로 로봇이 움직이지 않습니다.</p><div id="models"></div><div class="note"><label for="policy-select">GPU에 올라온 정책 중 선택</label><select id="policy-select" disabled></select><p class="sub">추론 종료 시 모델은 유지됩니다. 웹 종료 · GPU 해제 또는 터미널 Ctrl+C로 모두 내립니다.</p></div></section>
<section class="card"><h2>목표 변화량 제한</h2><p class="limit"><span id="limit">—</span> <small>rad / 정책 갱신</small></p><p class="sub" id="limit-scope">정책 선택 대기</p></section>
<section class="card"><h2>현재 실행</h2><dl><dt>모델</dt><dd id="checkpoint">—</dd><dt>GPU / PID</dt><dd id="gpu">—</dd><dt>복귀 자세</dt><dd id="pose">—</dd><dt>실제 명령 송신</dt><dd id="hardware">—</dd><dt>저장 위치</dt><dd id="output_dir">—</dd></dl></section>
</aside><main class="stack">
<section class="card"><div class="top"><span id="phase" role="status">서버 연결 중</span><span class="sub"><i class="dot"></i>100 Hz control</span></div>
<p class="sub" id="hint"></p><div class="buttons"><button data-action="start" disabled>Init Pose 이동 후 추론 시작</button><button data-action="pause" disabled>일시정지 → Init Pose</button><button data-action="resume" disabled>추론 재개</button><button data-action="stop" disabled>추론 종료</button></div><p id="error" role="alert"></p>
<div class="metrics"><div class="metric"><strong id="commands">0</strong><span>COMMANDS</span></div><div class="metric"><strong id="sequence">—</strong><span>POLICY SEQUENCE</span></div><div class="metric"><strong id="latency">—</strong><span>OBSERVATION → RESULT</span></div></div></section>
<section class="card" id="diagnostics-card" aria-label="추론 및 센서 진단">
<div class="top"><h2 style="margin:0">추론 · 입력 상태</h2><span id="velocity-gate" class="diag-pill muted">ACT 보간 상한 —</span></div>
<p class="sub">모델 재계산과 정책 결과를 따로 셉니다. ACT가 이전 재계산의 action chunk를 사용하면 두 Hz가 다를 수 있습니다.</p>
<div class="diagnostic-metrics">
<div class="metric"><strong id="model-forward-hz">—</strong><span>모델 재계산 Hz · 최근 2초</span></div>
<div class="metric"><strong id="policy-result-hz">—</strong><span>정책 결과 Hz · 최근 2초</span></div>
<div class="metric"><strong id="last-result-age">—</strong><span>마지막 새 결과 후 경과</span></div>
<div class="metric"><strong id="last-forward-age">—</strong><span>마지막 모델 재계산 후 경과</span></div>
</div>
<div class="diagnostic-badges"><span id="result-health" class="diag-pill muted">결과 대기</span><span id="model-health" class="diag-pill muted">모델 대기</span><span id="sender-health" class="diag-pill muted">명령 대기</span></div>
<p id="output-repeat" class="diagnostic-detail">출력 변화 측정 대기</p>
<p id="model-source" class="diagnostic-detail">재계산 여부 측정 대기</p>
<p id="sender-progress" class="diagnostic-detail">적용 결과 측정 대기</p>
<p id="sender-reason" class="diagnostic-detail" hidden></p>
<div class="sensor-table-wrap"><table class="sensor-table"><thead><tr><th>센서</th><th>사용한 새 프레임 Hz</th><th>최근 보고된 수신 Hz</th><th>해당 입력의 로컬 수신 나이</th><th>입력 프레임 ID</th><th>입력 상태</th><th>최근 재계산 사용</th></tr></thead><tbody id="sensor-diagnostics"><tr><td colspan="7">센서 진단 대기</td></tr></tbody></table></div>
<p class="sub">프레임 ID로 새 입력 여부를 확인합니다. 장면이 정지해 픽셀이 같아도 새 프레임일 수 있습니다. 수신 시각은 카메라 촬영 시각과 다릅니다. 모니터링용 센서는 모델 입력에서 제외됩니다. 알 수 없는 값은 —로 표시합니다.</p>
</section>
<section class="card" id="correction-card" aria-label="관절 목표 보정 로그">
<div class="top"><h2 style="margin:0">관절 목표 보정 로그</h2><span id="correction-current" class="diag-pill muted">보정 수신 대기</span></div>
<p id="correction-current-warning" class="diagnostic-detail correction-warning" hidden></p>
<p class="sub">ACT 기준: chunk → temporal ensemble → 역정규화 → 보정. 미래 chunk 전체가 아니라 실행기에 전달된 한 시점의 관절 목표를 비교합니다.</p>
<div class="top"><span id="correction-event-state" class="diag-pill muted">보정 기록 없음</span><span id="correction-count" class="sub">누적 보정 0건</span></div>
<p id="correction-event-info" class="diagnostic-detail">보정이 발생하면 마지막 기록을 유지합니다.</p>
<p id="correction-summary" class="diagnostic-detail" hidden></p>
<p id="correction-event-warning" class="diagnostic-detail correction-warning" hidden></p>
<div class="correction-table-wrap"><table class="sensor-table correction-table"><thead><tr><th>관절 · SDK 축</th><th>보정 전 (rad)</th><th>보정 후 (rad)</th><th>Δ (rad)</th><th>채택된 궤적 목표 (rad)</th><th>이벤트 시 100Hz 명령 (rad)</th></tr></thead><tbody id="correction-targets"><tr><td colspan="6">보정 기록 없음</td></tr></tbody></table></div>
<p class="sub">노란 행이 보정된 관절입니다. 보정 목표가 궤적 검사에서 거부되면 기존 궤적을 유지합니다. 수치는 이벤트 시점의 목표·명령이며 로봇 측정 각도가 아닙니다. 오류가 나면 마지막 수락 목표의 기록이 남습니다.</p>
<details class="correction-history-details"><summary id="correction-history-title">최근 보정 이력 · 최대 30건</summary><ol id="correction-history" class="correction-history"><li>보정 기록 없음</li></ol></details>
</section>
<section class="card"><div class="top"><h2 style="margin:0">Model observations</h2><span id="feed-state" class="badge stale">입력 대기</span></div>
<p class="sub">모델 입력은 정규화 전 RGB입니다. 좌우 tactile은 모든 정책에서 표시하며, 입력에 포함되지 않으면 모델 미사용으로 표시합니다.</p><div id="images" class="images"></div>
<p id="frame-info" class="sub note">추론을 시작하면 선택한 정책의 이미지가 표시됩니다.</p></section>
<p class="sub">추론 종료는 해당 rollout의 송신을 종료합니다. 브라우저 탭을 닫아도 서버와 모델은 유지됩니다. 비상정지나 Torque OFF 명령이 아닙니다.</p>
</main></div>
<script>
const token='__TOKEN__',put=(id,v)=>document.getElementById(id).textContent=v;
const labels={loading:'GPU 모델 확인 중',idle:'시작 대기',starting:'센서 · 추론 준비 중',policy:'추론 실행 중',braking:'추론 일시정지 · 감속 중',ready_gains:'자세 복귀 준비 중',moving:'Init Pose 이동 중',settling:'자세 도착 확인 중',waiting:'Init Pose 유지 · 재개 대기',restore_gains:'추론 재개 준비 중',resuming:'새 관측으로 추론 중',finished:'실행 종료 · 다음 정책 선택 가능',error:'오류로 중단',shutting_down:'추론 정리 · GPU 해제 중'};
const names={head_cam:'Ego camera',left_cam:'Left wrist',right_cam:'Right wrist',usb_tactile_left_heatmap:'Left tactile',usb_tactile_right_heatmap:'Right tactile',tactile_left_heatmap:'Left tactile',tactile_right_heatmap:'Right tactile'};
const statuses={unavailable:'체크포인트 파일 없음',unloaded:'GPU에 없음',unloading:'GPU 해제 중',loading:'GPU 로딩 중',ready:'GPU 상주 · 시작 시 세션 확인',busy:'GPU 상주 · 다른 실행에서 사용 중',active:'GPU 상주 · 이 웹에서 실행 중',error:'모델 서버 오류'};
const buttons=[...document.querySelectorAll('button[data-action]')],select=document.getElementById('policy-select');let imageKey='',frameKey='',modelKey='',optionsKey='',selectionPending=false,shutdownRequested=false;

const resultLabels={fresh:'새 결과 수신 중',stale:'새 결과 지연',waiting:'결과 대기',inactive:'추론 비활성'};
const modelLabels={fresh_forward:'모델 새 재계산 완료',cached_action_chunk:'이전 재계산의 action chunk 사용',unknown:'재계산 여부 미확인',stale:'새 모델 결과 지연',inactive:'모델 비활성'};
const senderLabels={applying:'새 결과 적용 중',same_goal:'같은 목표 · 기존 궤적 유지',retained_trajectory:'새 결과 적용 거부 · 기존 궤적 유지',stale_hold:'새 결과 없음 · 이전 명령 유지',first_frame_bridge:'첫 명령 연결 궤적 실행',inactive:'명령 비활성',unknown:'적용 상태 미확인'};
const sensorLabels={fresh:'새 프레임',reused:'동일 프레임 재사용',stale:'수신 지연',unknown:'식별 정보 없음',sample:'고정 샘플',inactive:'비활성'};
const sourceLabels={model_forward:'새 모델 재계산',fresh_forward:'새 모델 재계산',forward:'새 모델 재계산',cached_action_chunk:'저장된 action chunk',action_chunk:'저장된 action chunk',cached_chunk:'저장된 action chunk',unknown:'미확인'};
const diagNumber=(value,digits=1)=>Number.isFinite(value)?value.toFixed(digits):'—';
const diagAge=value=>Number.isFinite(value)?Math.round(value)+' ms':'—';
const diagCount=value=>Number.isFinite(value)?String(value):'—';
function diagnosticPill(id,text,tone){const node=document.getElementById(id);node.textContent=text;node.className='diag-pill '+tone;}
function drawDiagnostics(s){
 const d=s.diagnostics||{},sender=d.sender||{},active=s.phase==='policy';
 put('model-forward-hz',diagNumber(d.model_forward_hz));put('policy-result-hz',diagNumber(d.result_hz));
 put('last-result-age',diagAge(d.last_result_age_ms));put('last-forward-age',diagAge(d.last_forward_age_ms));
 const gates=['velocity_limit_gate','acceleration_limit_gate','jerk_limit_gate'].map(key=>typeof s[key]==='boolean'?s[key]:sender[key]);
 const gateText=gates.map((gate,index)=>['속도','가속도','jerk'][index]+' '+(typeof gate==='boolean'?(gate?'ON':'OFF'):'—')).join(' · ');
 diagnosticPill('velocity-gate','ACT 보간 상한: '+gateText,gates.some(gate=>gate===false)?'warn':'muted');
 const resultState=d.result_status||'waiting',modelState=d.model_status||'unknown',senderState=sender.status||'unknown';
 diagnosticPill('result-health',resultLabels[resultState]||'결과 상태 미확인',resultState==='fresh'?'good':resultState==='stale'?'bad':'muted');
 diagnosticPill('model-health',modelLabels[modelState]||'재계산 여부 미확인',modelState==='fresh_forward'?'good':modelState==='stale'?'bad':modelState==='cached_action_chunk'?'warn':'muted');
 diagnosticPill('sender-health',senderLabels[senderState]||'적용 상태 미확인',['retained_trajectory','stale_hold'].includes(senderState)?'bad':senderState==='applying'?'good':['same_goal','first_frame_bridge'].includes(senderState)?'warn':'muted');
 let repeat='출력 변화 측정 대기';
 if(Number.isFinite(d.max_action_delta_rad)&&d.output_exact_same===true)repeat='최근 결과와 값이 완전히 같음';
 else if(Number.isFinite(d.max_action_delta_rad)&&d.output_near_same===true)repeat='최근 결과와 값이 거의 같음';
 else if(Number.isFinite(d.max_action_delta_rad)&&d.output_exact_same===false)repeat='최근 결과와 출력값이 다름';
 if(Number.isFinite(d.max_action_delta_rad))repeat+=' · 연속 반복 '+diagCount(d.output_repeat_streak)+'회 · 최대 변화 '+diagNumber(d.max_action_delta_rad,6)+' rad';
 put('output-repeat',repeat+(d.output_exact_same===true||d.output_near_same===true?' · 동일한 값만으로 추론 중단을 뜻하지 않습니다.':''));
 const source=sourceLabels[d.output_source]||d.output_source||'미확인';
 put('model-source','결과 출처 '+source+' · 결과 '+diagCount(d.result_count)+'회 / 모델 재계산 '+diagCount(d.forward_count)+'회 · 결과 번호 '+diagCount(d.seq));
 put('sender-progress','받은 결과 #'+diagCount(sender.policy_seq)+' · 적용 결과 #'+diagCount(sender.applied_seq)+' · 받은 후 '+diagAge(sender.accepted_result_age_ms)+' / 적용 후 '+diagAge(sender.applied_age_ms)+' · 지연 결과 폐기 '+diagCount(sender.stale_results_dropped)+'회');
 const reason=document.getElementById('sender-reason');reason.hidden=!['retained_trajectory','stale_hold'].includes(senderState);
 reason.textContent=senderState==='retained_trajectory'?'새 추론 결과는 오고 있지만 궤적 제한 검사를 통과하지 못해 기존 궤적을 유지합니다. 적용 거부 '+diagCount(sender.replan_rejections)+'회 · 목표 차이 '+diagNumber(sender.goal_gap_rad,6)+' rad'+(sender.last_replan_error?'\\n'+sender.last_replan_error:''):'새 결과가 없어 이전 명령을 유지하고 있습니다. 마지막 수락 결과 후 '+diagAge(sender.accepted_result_age_ms);
 const rows=(d.sensors||[]).map(sensor=>{
  const row=document.createElement('tr'),status=sensor.status||'unknown',monitorOnly=sensor.model_input===false;
  const frame=sensor.frame_id==null?'—':typeof sensor.frame_id==='object'?JSON.stringify(sensor.frame_id):String(sensor.frame_id);
  const inputStatus=(monitorOnly?'모니터링 · ':'')+(sensorLabels[status]||'미확인')+(sensor.frame_reused===true?' · 누적 '+diagCount(sensor.reuse_count)+'회':'');
  const label=(names[sensor.name]||sensor.name||'센서')+(sensor.model_input===false?' · 모니터링용':'');
  const values=[label,monitorOnly?'—':diagNumber(sensor.input_hz),diagNumber(sensor.receive_hz),diagAge(sensor.receive_age_ms),frame,inputStatus,monitorOnly?'모델 미사용':typeof sensor.used_on_last_forward==='boolean'?(sensor.used_on_last_forward?'사용':'미사용'):'—'];
  values.forEach((value,index)=>{const cell=document.createElement('td');cell.textContent=value;if(index===4)cell.className='frame-id';if(index===5)cell.style.color=monitorOnly?'#91a1b8':status==='stale'?'#ffb8b8':status==='reused'?'#edbd7c':status==='fresh'?'#7fe0bb':'#91a1b8';row.append(cell);});return row;
 });
 if(!rows.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=7;cell.textContent=active?'센서 진단 수신 대기':'센서 진단 대기';row.append(cell);rows.push(row);}
 document.getElementById('sensor-diagnostics').replaceChildren(...rows);
}
function diagnosticsDisconnected(){
 for(const id of ['model-forward-hz','policy-result-hz','last-result-age','last-forward-age'])put(id,'—');
 diagnosticPill('result-health','웹 연결 끊김 · 측정 불가','bad');diagnosticPill('model-health','재계산 상태 미확인','muted');diagnosticPill('sender-health','명령 상태 미확인','muted');
 put('output-repeat','연결이 끊겨 최신 출력값을 확인할 수 없습니다.');put('model-source','표시된 과거 결과로 현재 추론 여부를 판단할 수 없습니다.');put('sender-progress','적용 상태 측정 불가');document.getElementById('sender-reason').hidden=true;
 const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=7;cell.textContent='웹 연결 끊김 · 센서 측정 불가';row.append(cell);document.getElementById('sensor-diagnostics').replaceChildren(row);
 correctionsDisconnected();
}
const correctionStates={applied:['보정 목표를 궤적에 채택','good'],unchanged_goal:['동일한 목표 · 기존 궤적 유지','muted'],rejected:['새 보정 목표 거부 · 이전 궤적 유지','bad'],waiting:['보정 목표 대기 · 채택 여부 미확인','warn'],stale_hold:['입력 지연 · 마지막 명령 유지','warn']};
const correctionRad=value=>Number.isFinite(value)?(value===0?0:value).toFixed(6):'—';
const correctionDelta=value=>Number.isFinite(value)?(value>0?'+':'')+correctionRad(value):'—';
const correctionCount=value=>Number.isFinite(value)?String(value):'—';
const correctionTime=value=>Number.isFinite(value)?value.toFixed(3)+' s':'—';
let correctionRenderKey=null;
function correctionState(event){return event.holding?['입력 지연 · 마지막 명령 유지','warn']:(correctionStates[event.replan_status]||['보정 목표 채택 여부 미확인','muted']);}
function correctionChanges(event){return (event.rows||[]).filter(axis=>(event.changed_axes||[]).includes(axis.axis)).map(axis=>(axis.name||'관절')+' [SDK '+axis.axis+'] '+correctionRad(axis.requested)+' → '+correctionRad(axis.corrected)+' (Δ '+correctionDelta(axis.delta)+' rad)').join('\\n');}
function correctionWarnings(event){return (event.correction_warnings||[]).map(warning=>(warning.name||'관절')+' [SDK '+warning.axis+'] 필요 '+correctionRad(warning.correction_rad)+' > 경고 기준 '+correctionRad(warning.threshold_rad)+' rad · '+correctionRad(warning.requested)+' → '+correctionRad(warning.corrected)+' rad').join('\\n');}
function drawCorrections(s){
 const log=s.correction_log||{},latest=log.latest||null,current=log.current||null,events=log.events||[];
 let currentText='보정 수신 대기',currentTone='muted';
 if(s.phase==='error'){currentText='중단됨 · 마지막 수락 목표의 보정 기록';currentTone='bad';}
 else if(s.phase!=='policy'&&latest){currentText='실행 비활성 · 마지막 보정 기록';}
 else if(current&&current.holding){currentText='입력 지연 · 마지막 명령 유지';currentTone='warn';}
 else if(current&&current.projected===true){const [text,tone]=correctionState(current);currentText='현재 목표 보정됨 · '+text;currentTone=tone;}
 else if(current&&current.projected===false){currentText='현재 목표 보정 없음';currentTone='good';}
 else if(latest){currentText='현재 목표 확인 대기 · 마지막 보정 기록';}
 diagnosticPill('correction-current',currentText,currentTone);
 const warnings=current&&current.correction_warnings||[],currentWarning=document.getElementById('correction-current-warning');
 currentWarning.hidden=s.phase!=='policy'||!warnings.length;
 currentWarning.textContent=warnings.length?'현재 보정량 경고 · '+warnings.length+'축 · 보정량만으로 중단하지 않음':'';
 put('correction-count','누적 보정 '+correctionCount(log.total_events==null?0:log.total_events)+'건 · 최근 '+events.length+'건');
 put('correction-history-title','최근 보정 이력 · 최대 '+correctionCount(log.capacity==null?30:log.capacity)+'건');
 const key=JSON.stringify([log.total_events,latest&&latest.id,latest&&latest.at,latest&&latest.policy_id,latest&&latest.generation,latest&&latest.policy_seq,latest&&latest.replan_status]);
 if(key===correctionRenderKey)return;correctionRenderKey=key;
 if(!latest){
  diagnosticPill('correction-event-state','보정 기록 없음','muted');put('correction-event-info','보정이 발생하면 마지막 기록을 유지합니다.');
  put('correction-summary','');document.getElementById('correction-summary').hidden=true;
  put('correction-event-warning','');document.getElementById('correction-event-warning').hidden=true;
  const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=6;cell.textContent='보정 기록 없음';row.append(cell);document.getElementById('correction-targets').replaceChildren(row);
  const item=document.createElement('li');item.textContent='보정 기록 없음';document.getElementById('correction-history').replaceChildren(item);return;
 }
 const [eventText,eventTone]=correctionState(latest);diagnosticPill('correction-event-state',eventText,eventTone);
 put('correction-event-info','보정 기록 +'+correctionTime(latest.elapsed_s)+' · 결과 #'+correctionCount(latest.policy_seq)+' · 세대 '+correctionCount(latest.generation)+' · 채택 결과 #'+correctionCount(latest.applied_seq)+' · '+(latest.policy_id==null?'정책 미지정':latest.policy_id)+' · 최대 보정 '+correctionRad(latest.max_correction_rad)+' rad');
 put('correction-summary',correctionChanges(latest));document.getElementById('correction-summary').hidden=false;
 const eventWarning=document.getElementById('correction-event-warning'),warningText=correctionWarnings(latest);
 eventWarning.hidden=!warningText;eventWarning.textContent=warningText?'이 보정 기록의 경고 · 보정량만으로 중단하지 않음\\n'+warningText:'';
 const changed=new Set(latest.changed_axes||[]);
 document.getElementById('correction-targets').replaceChildren(...(latest.rows||[]).map(axis=>{
  const row=document.createElement('tr');row.className=changed.has(axis.axis)?'changed':'unchanged';
  const values=[(axis.name||'관절')+' · SDK '+axis.axis+(changed.has(axis.axis)?' · 보정':' · 유지'),correctionRad(axis.requested),correctionRad(axis.corrected),correctionDelta(axis.delta),correctionRad(axis.trajectory_target),correctionRad(axis.command)];
  for(const value of values){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}return row;
 }));
 document.getElementById('correction-history').replaceChildren(...events.map(event=>{
  const item=document.createElement('li'),title=document.createElement('div'),values=document.createElement('div');
  title.textContent='+'+correctionTime(event.elapsed_s)+' · 결과 #'+correctionCount(event.policy_seq)+' / 세대 '+correctionCount(event.generation)+' · '+correctionState(event)[0];
  values.className='correction-log-values';values.textContent=correctionChanges(event);
  item.append(title,values);
  const warningText=correctionWarnings(event);
  if(warningText){const warning=document.createElement('div');warning.className='correction-log-warning';warning.textContent='보정량 경고 · '+warningText;item.append(warning);}return item;
 }));
}
function correctionsDisconnected(){diagnosticPill('correction-current','웹 연결 끊김 · 마지막 보정 기록','bad');document.getElementById('correction-current-warning').hidden=true;}
function draw(s){
 drawDiagnostics(s);
 drawCorrections(s);
 put('phase',labels[s.phase]||s.phase);put('mode',s.input_mode+' / '+s.output_mode);put('limit',s.max_action_step??'—');
 put('limit-scope',s.waist_yaw_step_gate===false?'waist yaw 제외 · 나머지 관절에 적용':'선택한 모든 관절에 적용');
 put('hint',s.input_mode==='sample'?'저장된 샘플 입력 · 로봇 명령 없음':s.output_mode==='shadow'?'Live shadow · 실물 관측 수신 · 로봇 명령 없음':'실제 로봇 명령 송신 모드');
 for(const id of ['checkpoint','output_dir'])put(id,s[id]);put('gpu',s.device+' / '+(s.server_pid||'—'));put('pose',s.pose_name);
 put('commands',s.command_count||0);put('sequence',s.policy_seq??'—');put('hardware',String(s.hardware_executed)+' · DDS '+s.dds_writes);
 buttons.forEach(b=>b.disabled=selectionPending||!s.actions.includes(b.dataset.action));put('error',s.error||s.preview_error||'');
 const policies=s.policies||[];document.getElementById('selector-card').hidden=!policies.length;document.getElementById('shutdown').hidden=!policies.length;
 shutdownRequested=shutdownRequested||s.phase==='shutting_down';
 const available=policies.filter(p=>['ready','active'].includes(p.status)),nextOptions=JSON.stringify(available.map(p=>[p.id,p.name]));
 if(nextOptions!==optionsKey){optionsKey=nextOptions;select.replaceChildren(new Option(available.length?'정책을 선택하세요':'먼저 모델을 GPU에 올리세요',''),...available.map(p=>new Option(p.name,p.id)));}
 if(!selectionPending)select.value=s.policy_id||'';select.disabled=selectionPending||!s.can_select||!available.length;
 const nextModels=JSON.stringify(policies);
 if(nextModels!==modelKey){modelKey=nextModels;document.getElementById('models').replaceChildren(...policies.map(p=>{
 const d=document.createElement('div');d.className='model';const n=document.createElement('div');n.className='model-name';n.textContent=p.name;
 const t=document.createElement('div');t.className='model-state';t.textContent=(statuses[p.status]||p.status)+(p.pid?' · PID '+p.pid:'');d.append(n,t);
 if(p.error){const e=document.createElement('div');e.className='model-error';e.textContent=p.error;d.append(e);}
 const actions=document.createElement('div');actions.className='model-buttons';for(const action of p.model_actions||[]){const b=document.createElement('button');b.textContent=action==='load'?'GPU에 올리기':'GPU에서 내리기';b.setAttribute('aria-label',p.name+' '+b.textContent);b.onclick=()=>act('/model',{action,policy_id:p.id});actions.append(b);}d.append(actions);return d;}));}
 document.querySelectorAll('.model-buttons button').forEach(b=>b.disabled=selectionPending);
 const current=policies.find(p=>p.id===s.policy_id),obs=s.observation;
 const schema=obs?obs.images.map(i=>[i.name,[i.height,i.width,3],i.model_input!==false]):Object.entries(current?.preview_images||current?.images||{}).map(([name,shape])=>[name,shape,name in(current?.images||{})]);
 const key=JSON.stringify([s.policy_id,schema]);if(key!==imageKey){imageKey=key;frameKey='';document.getElementById('images').replaceChildren(...schema.map(([name,shape,modelInput])=>{const f=document.createElement('figure'),a=document.createElement('div');a.className='image-area';const e=document.createElement('span');e.className='empty';e.textContent='관측 수신 대기';const img=document.createElement('img');img.alt=names[name]||name;img.dataset.name=name;img.hidden=true;a.append(e,img);const cap=document.createElement('figcaption'),title=document.createElement('span'),size=document.createElement('span');title.textContent=(names[name]||name)+' · '+(modelInput?'모델 입력':'모니터링용 · 모델 미사용');size.textContent=shape[1]+' × '+shape[0];cap.append(title,size);f.append(a,cap);return f;}));}
 if(obs){const next=JSON.stringify([obs.seq,obs.images.map(i=>i.url)]);if(next!==frameKey){for(const i of obs.images){const img=[...document.querySelectorAll('figure img')].find(x=>x.dataset.name===i.name);if(img){if(!i.url){img.onload=null;img.hidden=true;img.removeAttribute('src');img.previousElementSibling.hidden=false;img.previousElementSibling.textContent=s.input_mode==='sample'?'샘플에 tactile 데이터 없음':'tactile 수신 대기';continue;}img.onload=()=>{img.hidden=false;img.previousElementSibling.hidden=true;};img.src=i.url+'?seq='+obs.seq;}}frameKey=next;}put('feed-state',obs.stale?'마지막 정책 입력':'정책 입력 표시 중');document.getElementById('feed-state').classList.toggle('stale',obs.stale);put('frame-info','정책 결과 #'+obs.seq+' · 입력 기준 나이 '+Math.round(obs.age_ms)+' ms · 정책 호출 '+obs.inference_ms.toFixed(1)+' ms');put('latency',obs.result_latency_ms.toFixed(1)+' ms');}
 else{document.querySelectorAll('figure img').forEach(img=>{img.onload=null;img.hidden=true;img.removeAttribute('src');img.previousElementSibling.hidden=false;img.previousElementSibling.textContent='관측 수신 대기';});frameKey='';put('feed-state','입력 대기');document.getElementById('feed-state').classList.add('stale');put('frame-info','추론을 시작하면 선택한 정책의 입력과 좌우 tactile이 표시됩니다.');put('latency','—');}
}
async function send(path,data){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-ACT-Token':token},body:JSON.stringify(data)});const s=await r.json();if(!r.ok)throw Error(s.error);return s;}
async function act(path,data){selectionPending=true;document.querySelectorAll('button').forEach(b=>b.disabled=true);select.disabled=true;try{const s=await send(path,data);selectionPending=false;draw(s);}catch(e){put('error',e.message);}finally{selectionPending=false;}}
select.onchange=()=>{if(select.value)act('/policy',{policy_id:select.value});};
buttons.forEach(b=>b.onclick=()=>act('/command',{action:b.dataset.action}));
async function poll(){let delay=200;try{const r=await fetch('/status');if(!r.ok)throw Error('서버 응답 오류');const s=await r.json();draw(s);if(s.phase==='policy')delay=50;}catch(e){put('phase',shutdownRequested?'종료 요청 완료 · 서버 연결 종료':'서버 연결 끊김');document.querySelectorAll('button').forEach(b=>b.disabled=true);select.disabled=true;diagnosticsDisconnected();put('feed-state','연결 끊김 · 마지막 프레임');document.getElementById('feed-state').classList.add('stale');}setTimeout(poll,delay);}poll();
</script></body></html>
"""
