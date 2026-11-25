"""Browser controls for the existing recorder; camera reads stay in record_loop."""

import copy
import json
import queue
import re
import secrets
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from lerobot.datasets.record_phases import PHASE_FEATURE, PHASE_KEY, PHASE_LABELS
from lerobot.utils.cycle_timer import CycleTimer

from .record_handoff import read_packet

DEFAULT_CUMULATIVE_REPO = "kimyg119/Box_Palletizing_Scenario-A_Bucheon_Warehouse"
DEFAULT_OUTPUT_DIR = Path("/media/rcv/T5_EVO/dataset/igris")


class RecordWebControl:
    def __init__(
        self,
        cfg,
        output,
        *,
        host="127.0.0.1",
        port=8129,
        init_pose=None,
        poses=None,
        preview_only=False,
        cumulative_repo_id=DEFAULT_CUMULATIVE_REPO,
    ):
        if (
            cfg.robot.type != "igris_c"
            or cfg.robot.allow_control
            or cfg.teleop is None
            or cfg.teleop.type != "igris_c"
            or not cfg.teleop.require_lowcmd
        ):
            raise ValueError(
                "record web requires passive igris_c recording with igris_c teleop and require_lowcmd=true"
            )
        self.cfg, self.output, self.init_pose = cfg, Path(output).resolve(), init_pose
        self.poses, self.preview_only = poses, preview_only
        self.cumulative_repo_id = cumulative_repo_id
        self.lock = threading.RLock()
        self.events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
        self.requests = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.pose_stop = threading.Event()
        self.pose_thread = None
        self.upload_thread = None
        self.failed_upload = None
        self.pending = None
        self.episode_generation = 0
        self.phase_handoff = None
        self.token = secrets.token_urlsafe(32)
        self.state = {
            "phase": "idle",
            "record_phase": 0,
            "record_phase_labels": PHASE_LABELS,
            "saved_episodes": 0,
            "frames": 0,
            "discarded_episodes": 0,
            "target_episodes": cfg.dataset.num_episodes,
            "fps": cfg.dataset.fps,
            "igris_warehouse_schema": cfg.igris_warehouse_schema,
            "episode_time_s": cfg.dataset.episode_time_s,
            "error": None,
            "output_dir": None,
            "output_root": str(self.output),
            "repo_id": None,
            "upload_phase": "idle",
            "hub_repo_id": cumulative_repo_id or None,
            "hub_revision": None,
            "hub_total_episodes": None,
            "hub_total_frames": None,
            "hub_url": None,
            "hub_private": True if cfg.dataset.private is None else cfg.dataset.private,
            "init_phase": "idle",
            "init_error": None,
            "hardware_executed": False,
            "dds_write_attempts": 0,
            "dds_writes": 0,
            "defaults": {
                "name": cfg.dataset.repo_id.split("/")[-1],
                "num_episodes": cfg.dataset.num_episodes,
                "instruction": cfg.dataset.single_task,
            },
            "pose_name": getattr(init_pose, "name", None),
            "pose_steps": poses.descriptors() if poses else [],
            "pose_action": None,
            "pose_segment": 0,
            "pose_segments": 0,
            "tow_phase": "Viser LowCmd/TOW3를 한 번 연결하세요",
            "preview_only": preview_only,
            "shoulder_lift_deg": poses.data["shoulder_lift_deg"] if poses else None,
        }
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, data, content_type="application/json"):
                body = (
                    data
                    if isinstance(data, bytes)
                    else (
                        data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
                    )
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def same_origin(self):
                port = self.server.server_port
                host = self.headers.get("Host")
                return host in {
                    f"127.0.0.1:{port}",
                    f"localhost:{port}",
                    f"{self.connection.getsockname()[0]}:{port}",
                } and self.headers.get("Origin") in (None, f"http://{host}")

            def do_GET(self):  # noqa: N802
                if not self.same_origin():
                    self.reply(403, {"error": "same origin required"})
                elif self.path == "/":
                    page = Path(__file__).with_name("record_web.html").read_text()
                    self.reply(200, page.replace("__TOKEN__", control.token), "text/html; charset=utf-8")
                elif self.path == "/status":
                    self.reply(200, control.snapshot())
                else:
                    self.reply(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if not self.same_origin() or not secrets.compare_digest(
                    self.headers.get("X-Record-Token", "").encode(), control.token.encode()
                ):
                    self.reply(403, {"error": "same-origin page token required"})
                    return
                if self.path != "/command":
                    self.reply(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 8192:
                        raise ValueError("invalid command length")
                    self.connection.settimeout(2)
                    control.command(json.loads(self.rfile.read(length)))
                except (ValueError, TypeError, OSError) as exc:
                    self.reply(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self.reply(409, {"error": str(exc)})
                else:
                    self.reply(200, control.snapshot())

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def actions(self):
        if self.closed.is_set() or self.preview_only or self.state["phase"] == "uploading":
            return []
        phase = self.state["phase"]
        if self.state["init_phase"] not in ("idle", "complete", "error"):
            return ["pose_stop", "stop"] if self.state["pose_action"] else ["stop"]
        pose_actions = ["pose"] if self.poses and self.init_pose else []
        if phase in ("idle", "complete", "error"):
            return (
                ["start"]
                + (["retry_upload"] if self.failed_upload else [])
                + (["init_pose"] if self.init_pose else [])
                + pose_actions
            )
        if phase == "recording":
            return ["next", "reset", "stop"] + pose_actions
        if phase == "ready":
            return ["next", "stop"] + (["init_pose"] if self.init_pose else []) + pose_actions
        if phase in ("connecting", "ending_episode", "saving", "discarding"):
            return ["stop"]
        return []

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result["actions"] = self.actions()
        return result

    def update(self, *, episode_generation=None, phase_handoff=None, **values):
        with self.lock:
            if episode_generation is not None and episode_generation != self.episode_generation:
                return
            self.state.update(values)
            if self.state["phase"] == "recording":
                if phase_handoff is not None and self.state["record_phase"] in (1, 3, 4):
                    self.phase_handoff = phase_handoff
                if (
                    self.state["pose_action"] == "post_grasp_place"
                    and self.state["record_phase"] == 3
                    and values.get("pose_segment", 0) >= 2
                ):
                    self.state["record_phase"] = 4
            if values.get("init_phase") == "error":
                self.phase_handoff = None

    @property
    def annotation_features(self):
        return {PHASE_KEY: copy.deepcopy(PHASE_FEATURE)}

    def frame_annotations(self):
        with self.lock:
            handoff = self.phase_handoff
            if handoff is not None and not self.pose_stop.is_set() and self.state["record_phase"] in (1, 4):
                status = read_packet(Path(handoff["directory"]) / "status.json")
                stamp, reply = status.get("at"), status.get("reply") or {}
                if (
                    type(stamp) in (int, float)
                    and 0 <= time.monotonic() - stamp <= 2
                    and status.get("active") is True
                    and status.get("phase") == "enabled"
                    and status.get("session_status") == "TOW3_CONTINUOUS_TRACKING"
                    and isinstance(reply, dict)
                    and reply.get("id") == handoff["id"]
                    and reply.get("lease") == handoff["lease"]
                    and reply.get("ok") is True
                    and reply.get("phase") == "enabled"
                ):
                    self.state["record_phase"] += 1
                    self.phase_handoff = None
            return {PHASE_KEY: np.array([self.state["record_phase"]], dtype=np.int64)}

    def command(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("action"), str):
            raise ValueError("expected an action object")
        action = data["action"]
        required = {"action", "name", "num_episodes", "instruction"} if action == "start" else {"action"}
        if action == "pose":
            required.add("pose")
        if set(data) != required:
            raise ValueError(f"expected {sorted(required)}")
        if action not in ("start", "next", "reset", "stop", "init_pose", "pose", "pose_stop", "retry_upload"):
            raise ValueError("unknown action")
        if action == "pose" and (
            not self.poses or data["pose"] not in [p["id"] for p in self.state["pose_steps"]]
        ):
            raise ValueError("unknown pose")
        with self.lock:
            if action not in self.actions():
                raise RuntimeError(f"{action} unavailable during {self.state['phase']}")
            if (
                action in ("start", "init_pose", "pose")
                and self.output.is_relative_to(DEFAULT_OUTPUT_DIR.parents[1])
                and not DEFAULT_OUTPUT_DIR.parents[1].is_mount()
            ):
                raise RuntimeError("T5_EVO is not mounted; connect the recording drive before starting")
            if action == "start":
                name, count, instruction = data["name"], data["num_episodes"], data["instruction"]
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,71}", name):
                    raise ValueError("name: use 1–72 letters, digits, _ or -; begin with a letter or digit")
                if name.startswith("eval_") or "--" in name:
                    raise ValueError("name must not start with eval_ or contain --")
                if type(count) is not int or not 1 <= count <= 10000:
                    raise ValueError("num_episodes must be an integer from 1 to 10000")
                if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 2000:
                    raise ValueError("instruction must contain 1–2000 characters")
                cfg = copy.deepcopy(self.cfg)
                suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = f"{name}_{suffix}"
                owner = cfg.dataset.repo_id.partition("/")[0] if "/" in cfg.dataset.repo_id else "local"
                cfg.dataset.repo_id = f"{owner}/{dataset_name}"
                cfg.dataset.root = self.output / dataset_name
                cfg.dataset.no_stamp = True
                cfg.dataset.num_episodes = count
                cfg.dataset.single_task = instruction.strip()
                # The web runner uploads only after record() finishes successfully and releases sensors.
                cfg.dataset.push_to_hub = False
                if cfg.dataset.private is None:
                    cfg.dataset.private = True
                cfg.display_data = cfg.play_sounds = cfg.resume = False
                self.events.update(exit_early=False, stop_recording=False, rerecord_episode=False)
                self.pending = None
                self.failed_upload = None
                self.state.update(
                    phase="connecting",
                    saved_episodes=0,
                    frames=0,
                    discarded_episodes=0,
                    target_episodes=count,
                    error=None,
                    output_dir=str(cfg.dataset.root),
                    repo_id=cfg.dataset.repo_id,
                    upload_phase="pending" if self.cfg.dataset.push_to_hub else "skipped",
                    hub_repo_id=self.cumulative_repo_id or cfg.dataset.repo_id,
                    hub_revision=None,
                    hub_total_episodes=None,
                    hub_total_frames=None,
                    hub_url=None,
                    hub_private=cfg.dataset.private,
                    instruction=cfg.dataset.single_task,
                )
                self.requests.put_nowait(cfg)
            elif action == "retry_upload":
                self.state.update(phase="uploading", upload_phase="uploading", error=None)
                self.upload_thread = threading.Thread(target=self._retry_upload, daemon=True)
                self.upload_thread.start()
            elif action in ("init_pose", "pose"):
                self.pose_stop.clear()
                self.phase_handoff = None
                if self.state["phase"] == "recording" and self.state["record_phase"] != 6:
                    self.state["record_phase"] = {"pre_pick": 1, "post_grasp_place": 3, "return_home": 6}[
                        data["pose"]
                    ]
                self.state.update(
                    init_phase="checking",
                    init_error=None,
                    hardware_executed=False,
                    dds_write_attempts=0,
                    dds_writes=0,
                    pose_action=data.get("pose"),
                    pose_segment=0,
                    pose_segments=0,
                )
                self.pose_thread = threading.Thread(
                    target=self._move_to_pose,
                    args=(data.get("pose"), self.episode_generation),
                    daemon=True,
                )
                self.pose_thread.start()
            elif action == "pose_stop":
                self.pose_stop.set()
                self.phase_handoff = None
            elif action == "stop":
                self.pose_stop.set()
                if self.state["phase"] not in ("idle", "complete", "error"):
                    self.events.update(stop_recording=True, exit_early=True)
                    self.state["phase"] = "ending_episode"
            else:
                self.pending = action
                self.events["exit_early"] = True
                self.state["phase"] = "ending_episode"

    def _move_to_pose(self, action=None, episode_generation=None):
        def update(**values):
            self.update(episode_generation=episode_generation, **values)

        try:
            if action:
                self.init_pose.move(action, self.pose_stop, update)
            else:
                self.init_pose(self.pose_stop, update)
        except Exception as exc:
            update(init_phase="error", init_error=f"{type(exc).__name__}: {exc}")
        else:
            update(init_phase="complete")

    def _begin_phase(self, phase):
        with self.lock:
            if self.events["stop_recording"] or self.closed.is_set():
                return False
            self.pending = None
            self.events["exit_early"] = False
            self.state["phase"] = phase
            if phase == "recording":
                self.state["frames"] = 0
                self.state["record_phase"] = 1
                self.episode_generation += 1
                self.phase_handoff = None
            return True

    def record_episodes(self, loop, dataset, cfg, timer):
        """Next saves, Reset discards, and every following episode waits for Next."""
        self.update(output_dir=str(dataset.root), repo_id=dataset.repo_id)
        try:
            while self.state["saved_episodes"] < cfg.dataset.num_episodes and self._begin_phase("recording"):
                loop(dataset=dataset, control_time_s=cfg.dataset.episode_time_s, timer=timer)
                with self.lock:
                    discard = self.pending == "reset"
                    self.state["phase"] = "discarding" if discard else "saving"
                if discard:
                    dataset.clear_episode_buffer()
                    with self.lock:
                        self.state["discarded_episodes"] += 1
                elif dataset.has_pending_frames():
                    dataset.save_episode()
                    with self.lock:
                        self.state["saved_episodes"] += 1
                timer.log_episode_summary("discarded episode" if discard else "saved episode")
                timer.restart()
                if self.state["saved_episodes"] >= cfg.dataset.num_episodes or not self._begin_phase("ready"):
                    break
                loop(
                    control_time_s=float("inf"),
                    timer=CycleTimer(cfg.dataset.fps, records_data=False, warning_fps=cfg.warning_fps),
                )
        finally:
            self.update(phase="finalizing")

    def upload_dataset(self, dataset, cfg):
        self.pose_stop.set()
        if not self.cfg.dataset.push_to_hub or dataset.num_episodes == 0:
            self.update(upload_phase="skipped")
            return
        self.update(phase="uploading", upload_phase="uploading")
        try:
            if self.cumulative_repo_id:
                from lerobot.datasets.cumulative import append_dataset

                receipt = append_dataset(Path(dataset.root), self.cumulative_repo_id, private=cfg.private)
                self.update(
                    hub_revision=receipt["revision"],
                    hub_total_episodes=receipt["total_episodes"],
                    hub_total_frames=receipt["total_frames"],
                )
            else:
                dataset.push_to_hub(tags=cfg.tags, private=cfg.private)
        except Exception:
            self.failed_upload = (dataset, cfg)
            self.update(upload_phase="error")
            raise
        self.failed_upload = None
        repo_id = self.cumulative_repo_id or dataset.repo_id
        self.update(upload_phase="complete", hub_url=f"https://huggingface.co/datasets/{repo_id}")

    def _retry_upload(self):
        try:
            self.upload_dataset(*self.failed_upload)
        except Exception as exc:
            self.finish(f"{type(exc).__name__}: {exc}")
        else:
            self.finish()

    def finish(self, error=None):
        self.pose_stop.set()
        self.update(phase="error" if error else "complete", error=error)

    def observe(self, observation, action, recording):
        with self.lock:
            if recording:
                self.state["frames"] += 1

    def stop(self):
        self.closed.set()
        self.pose_stop.set()
        with self.lock:
            self.events.update(stop_recording=True, exit_early=True)

    def close(self):
        self.stop()
        if self.pose_thread:
            self.pose_thread.join()
        if self.upload_thread:
            self.upload_thread.join()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
