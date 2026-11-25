"""Send recording motions to Viser's existing Core/LowCmd owner."""

import contextlib
import fcntl
import json
import os
import threading
import time
import uuid
from pathlib import Path

KIND = "igris-record-handoff-v2"
DEFAULT_ROOT = Path(__file__).resolve().parents[6] / "igris_object_qp/artifacts/operator_events/record_handoff"


def read_packet(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_packet(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def discover(root):
    candidates = []
    old = False
    for path in Path(root).glob("*/status.json"):
        state = read_packet(path)
        stamp = state.get("at")
        if (
            type(stamp) in (float, int)
            and 0 <= time.monotonic() - stamp <= 2
            and state.get("eligible") is True
            and state.get("lease") is None
            and state.get("phase") != "closed"
        ):
            try:
                os.kill(int(state["pid"]), 0)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if state.get("kind") != KIND or state.get("persistent_motion") is not True:
                old = True
            else:
                candidates.append((path.parent, state))
    if old:
        raise RuntimeError("저장 모션을 같은 송신기에서 실행하도록 업데이트한 Viser로 재시작하세요")
    if len(candidates) != 1:
        raise RuntimeError(
            f"TOW 인계 가능한 Viser 세션이 {len(candidates)}개입니다. "
            "업데이트한 Viser에서 LowCmd/TOW3를 한 번 연결하고, 연결된 세션은 하나만 유지하세요."
        )
    return candidates[0]


class ViserTowHandoff:
    def __init__(self, root, output, stopped, update, *, auto_resume=False):
        self.root, self.output = Path(root), Path(output)
        self.stopped, self.update = stopped, update
        self.auto_resume = auto_resume
        self.lease = uuid.uuid4().hex
        self.directory = self.lock_file = self.thread = None
        self.finished = threading.Event()
        self.state = {}
        self.fault = None
        self.events = []
        self.owns_lock = False
        self.sent = False

    def _write(self, name, value):
        write_packet(self.directory / name, value)

    def _heartbeat(self):
        while not self.finished.is_set():
            try:
                self._write("lease.json", {"lease": self.lease, "at": time.monotonic()})
            except OSError as exc:
                self.fault = str(exc)
                return
            self.finished.wait(0.1)

    def check(self):
        if self.stopped.is_set():
            raise InterruptedError("operator stopped TOW/pose handoff")
        self.state = read_packet(self.directory / "status.json")
        stamp = self.state.get("at")
        if (
            self.fault
            or type(stamp) not in (float, int)
            or not 0 <= time.monotonic() - stamp <= 2
            or self.state.get("token") != self.token
            or self.state.get("phase") == "closed"
            or self.state.get("active") is not True
        ):
            raise RuntimeError(f"Viser 연결 중단: {self.fault or self.state.get('reply') or 'stale'}")

    def __enter__(self):
        try:
            self.directory, self.state = discover(self.root)
            self.token = self.state["token"]
            self.lock_file = (self.directory / "client.lock").open("a")
            try:
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("다른 Record Studio가 Viser 인계 중입니다") from exc
            self.owns_lock = True
            self._write("lease.json", {"lease": self.lease, "at": time.monotonic()})
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
            self.check()
            return self
        except BaseException:
            self._close()
            raise

    def execute(self, motion):
        self.check()
        identity = uuid.uuid4().hex
        command = dict(
            id=identity,
            lease=self.lease,
            token=self.token,
            action="motion",
            motion=motion,
            auto_resume=self.auto_resume,
            at=time.monotonic(),
        )
        self._write("command.json", command)
        self.sent = True
        self.update(phase_handoff={"directory": str(self.directory), "id": identity, "lease": self.lease})
        self.events.append(dict(action="motion", id=identity, at=command["at"]))
        self.update(init_phase="checking", tow_phase="Viser 연결 유지 · 저장 모션으로 전환 중")
        deadline = time.monotonic() + 95
        while True:
            self.check()
            reply = self.state.get("reply") or {}
            if not isinstance(reply, dict):
                raise RuntimeError("invalid Viser motion reply")
            progress = self.state.get("motion") or {}
            if progress.get("id") == identity and progress.get("lease") == self.lease:
                self.update(
                    **{
                        key: progress[key]
                        for key in (
                            "init_phase",
                            "pose_segment",
                            "pose_segments",
                            "hardware_executed",
                            "dds_writes",
                        )
                        if key in progress
                    }
                )
            if reply.get("id") == identity and reply.get("lease") == self.lease:
                if reply.get("ok") is not True:
                    raise RuntimeError(reply.get("error", "Viser motion rejected"))
                expected = "enabled" if self.auto_resume else "wait_c"
                if reply.get("phase") != expected or self.state.get("phase") != expected:
                    raise RuntimeError("Viser motion completion state mismatch")
                self.events.append(dict(action="motion_confirmed", at=time.monotonic()))
                self.update(
                    tow_phase="TOW 자동 추종 시작 요청 전달 · 오른쪽 C로 일시정지"
                    if self.auto_resume
                    else "WAIT_TOW3_ENABLE · 오른쪽 C로 추종 시작"
                )
                return
            if self.state.get("lease") not in (None, self.lease):
                raise RuntimeError("Viser motion lease changed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Viser motion confirmation timed out")
            self.stopped.wait(0.02)

    def _abort(self):
        if self.owns_lock and self.sent:
            with contextlib.suppress(OSError):
                self._write(
                    "command.json",
                    dict(
                        id=uuid.uuid4().hex,
                        lease=self.lease,
                        token=self.token,
                        action="abort",
                        at=time.monotonic(),
                    ),
                )
            self.update(tow_phase="모션 중단 · 자동 TOW 재개 없음")

    def _close(self):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.lock_file:
            self.lock_file.close()
        self.owns_lock = False
        self.output.mkdir(parents=True, exist_ok=True)
        write_packet(
            self.output / f"{self.lease}.json",
            dict(
                lease=self.lease,
                viser_directory=str(self.directory),
                events=self.events,
                final_phase=self.state.get("phase"),
                motion=self.state.get("motion"),
            ),
        )

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is not None:
                self._abort()
        finally:
            self._close()
