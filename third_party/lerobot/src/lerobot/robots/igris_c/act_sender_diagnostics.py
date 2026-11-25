"""Bounded command logging and phase timings, without robot IO."""

import json
from pathlib import Path
import queue
import threading
import time


class AsyncCommandLog:
    """Keep JSON encoding and filesystem writes off the 100 Hz sender thread.

    Records must own their arrays/lists once submitted. Overload or write failure
    is a latched error checked before each subsequent command, never a silent drop.
    Close only after the command writer has been detached.
    """

    def __init__(self, path, *, capacity=512):
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()
        self.error = None
        self.submitted = 0
        self.written = 0
        self.peak_pending = 0
        self.file = Path(path).open("x")
        self.thread = threading.Thread(target=self._consume, name="act-command-log", daemon=True)
        self.thread.start()

    def _consume(self):
        try:
            while not self.closed.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                self.file.write(json.dumps(record, allow_nan=False) + "\n")
                self.written += 1
            self.file.flush()
        except BaseException as exc:
            self.error = f"command log writer failed: {type(exc).__name__}: {exc}"
        finally:
            try:
                self.file.close()
            except BaseException as exc:
                self.error = self.error or f"command log close failed: {type(exc).__name__}: {exc}"

    def check(self):
        # One producer owns the sender; the consumer can only free queue slots.
        # Checking before DDS prevents sending a command with no log capacity.
        if self.error is None and self.queue.full():
            self.error = "command log queue full; sender stopped without dropping queued records"
        if self.error is not None:
            raise RuntimeError(self.error)

    def write(self, record):
        self.check()
        if self.closed.is_set():
            raise RuntimeError("command log is closed")
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.error = "command log queue full; sender stopped without dropping queued records"
            self.check()
        self.submitted += 1
        self.peak_pending = max(self.peak_pending, self.queue.qsize())

    def close(self, *, timeout=5.):
        self.closed.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            self.error = self.error or "command log shutdown timed out; queued records may remain"
        return {"submitted": self.submitted, "written": self.written,
                "pending": self.queue.qsize(), "peak_pending": self.peak_pending,
                "thread_alive": self.thread.is_alive(), "error": self.error}


class SenderTiming:
    """Record the failed tick as well as the previous completed tick."""

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.current = None
        self.previous = None
        self.failure = None
        self.peaks_ms = {}
        self.count = 0

    def begin(self, deadline, tick, phase):
        self.current = {"tick": tick, "control_phase": phase, "deadline": deadline, "stages_ms": {}}
        self.last = deadline

    def mark(self, stage):
        now = self.clock()
        elapsed = max(0., now - self.last) * 1000
        self.current["stages_ms"][stage] = elapsed
        self.peaks_ms[stage] = max(self.peaks_ms.get(stage, 0.), elapsed)
        self.last = now
        return now

    def check(self, stage, require_deadline, *, gate):
        now = self.mark(stage)
        try:
            return require_deadline(now, self.current["deadline"], gate=gate)
        except TimeoutError as exc:
            self.failure = {**self.current, "stages_ms": dict(self.current["stages_ms"]),
                            "failed_stage": stage, "lateness_ms": (now - self.current["deadline"]) * 1000,
                            "previous_tick": self.previous}
            detail = ", ".join(f"{key}={value:.2f}ms" for key, value in self.current["stages_ms"].items())
            raise TimeoutError(f"{exc}; stage={stage}; {detail}") from exc

    def finish(self):
        self.count += 1
        self.previous = self.current
        self.current = None

    def snapshot(self):
        return {"completed_ticks": self.count, "peak_stage_ms": dict(self.peaks_ms),
                "last_completed_tick": self.previous, "incomplete_tick": self.current,
                "deadline_failure": self.failure}
