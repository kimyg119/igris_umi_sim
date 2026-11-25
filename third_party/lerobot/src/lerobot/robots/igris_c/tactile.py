"""Headless adapter of Tactile 2.zip/multi_thread_contact.py.

Matches sketch_nov27a: AA55 + rows*(cols one-hot ADC + all-HIGH sum) uint16 LE.
Preserves the supplied denominator EMA, median baseline and fixed-scale delta-g.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _serial_available, require_package

from .ros2_tactile import Ros2Tactile, Ros2TactileConfig

if TYPE_CHECKING or _serial_available:
    import serial

logger = logging.getLogger(__name__)
ROWS, COLS, OVERSAMPLE = 8, 32, 16
ADC_FULL = 1023.0
HEADER = b"\xaa\x55"
FRAME_BYTES = ROWS * (COLS + 1) * 2


@dataclass
class SerialTactileConfig:
    port: str
    baud: int = 2_000_000
    init_frames: int = 30
    denom_min: float = 2.0
    denom_alpha: float = 0.3
    threshold_g: float = 0.05
    g_scale: float = 3.0
    heatmap_alpha: float = 0.2
    timeout_s: float = 1.0
    startup_timeout_s: float = 10.0
    rows: int = ROWS
    cols: int = COLS

    def __post_init__(self):
        for name in ("rows", "cols"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.rows, self.cols) != (ROWS, COLS):
            raise ValueError("Serial tactile firmware must emit 8x32")
        if not self.port or type(self.baud) is not int or self.baud <= 0:
            raise ValueError("Tactile requires a port and positive integer baud")
        if type(self.init_frames) is not int or self.init_frames < 1:
            raise ValueError("init_frames must be a positive integer")
        for name in ("denom_min", "g_scale", "timeout_s", "startup_timeout_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("denom_alpha", "heatmap_alpha"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.threshold_g) or self.threshold_g < 0:
            raise ValueError("threshold_g must be finite and non-negative")


def validate_tactile_sources(configs):
    sources = [
        ("ros2", c.domain_id if c.domain_id is not None else int(os.environ.get("ROS_DOMAIN_ID", 0)), c.topic)
        if isinstance(c, Ros2TactileConfig) else ("serial", c.port)
        for c in configs.values()
    ]
    if len(sources) != len(set(sources)):
        raise ValueError("Each tactile sensor requires a distinct serial port or ROS2 domain/topic")


def make_tactile_sensors(configs):
    validate_tactile_sources(configs)
    return {name: Ros2Tactile(c) if isinstance(c, Ros2TactileConfig) else SerialTactile(c)
            for name, c in configs.items()}


def frame_iter(device, stop: threading.Event, *, rows: int = ROWS, cols: int = COLS):
    """Resynchronize fragmented serial frames; reject values outside the firmware ADC range."""
    if (rows, cols) != (ROWS, COLS):
        raise ValueError("Serial tactile firmware must emit 8x32")
    buf = bytearray()
    frame_bytes = rows * (cols + 1) * 2
    while not stop.is_set():
        buf += device.read(min(device.in_waiting, 4096) or 1)
        while True:
            idx = buf.find(HEADER)
            if idx < 0:
                del buf[:-1]
                break
            if len(buf) < idx + 2 + frame_bytes:
                del buf[:idx]
                break
            data = np.frombuffer(bytes(buf[idx + 2:idx + 2 + frame_bytes]), dtype="<u2").reshape(rows, cols + 1)
            if np.any(data[:, :cols] > ADC_FULL) or np.any(data[:, cols] > ADC_FULL * OVERSAMPLE):
                del buf[:idx + 2]
                continue
            del buf[:idx + 2 + frame_bytes]
            yield data[:, :cols].astype(np.float64), data[:, cols].astype(np.float64) / OVERSAMPLE


class TactileProcessor:
    def __init__(self, config: SerialTactileConfig):
        self.config = config
        self._denom = None
        self._init = []
        self.baseline = None
        self._visual = np.zeros((config.rows, config.cols))

    def update(self, raw, raw_all):
        c = self.config
        if raw.shape != (ROWS, COLS) or raw_all.shape != (ROWS,):
            raise ValueError("Expected 8x32 ADC values and eight all-HIGH row values")
        denom = np.maximum(ADC_FULL - raw_all, c.denom_min)
        self._denom = denom if self._denom is None else c.denom_alpha * denom + (1 - c.denom_alpha) * self._denom
        g = raw / self._denom[:, None]
        if self.baseline is None:
            self._init.append(g)
            if len(self._init) == c.init_frames:
                self.baseline = np.median(self._init, axis=0)
                self._init.clear()
            return None
        norm = np.clip((g - self.baseline - c.threshold_g) / c.g_scale, 0.0, 1.0)
        # Display-only smoothing per received frame; raw retains contact_data_norm.
        self._visual = c.heatmap_alpha * norm + (1 - c.heatmap_alpha) * self._visual
        return norm.astype(np.float32), self._visual.astype(np.float32)


class SerialTactile:
    def __init__(self, config: SerialTactileConfig):
        self.config = config
        self._serial = None
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._received = 0.0
        self._error = None

    def connect(self):
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        self.disconnect()
        self._stop.clear()
        self._ready.clear()
        self._latest, self._error = None, None
        logger.info("Tactile %s: keep unloaded during %d baseline frames", self.config.port, self.config.init_frames)
        try:
            self._serial = serial.Serial(self.config.port, self.config.baud, timeout=0.1, exclusive=True)
            self._serial.reset_input_buffer()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            if not self._ready.wait(self.config.startup_timeout_s):
                raise TimeoutError(
                    f"Tactile {self.config.port}: no calibrated frame; "
                    f"expected {self.config.rows}x{self.config.cols} at {self.config.baud} baud; "
                    "check firmware dimensions/baud/port"
                )
            self.get()
        except BaseException:
            self.disconnect()
            raise

    def _read_loop(self):
        processor = TactileProcessor(self.config)
        try:
            for raw, raw_all in frame_iter(
                self._serial, self._stop, rows=self.config.rows, cols=self.config.cols
            ):
                maps = processor.update(raw, raw_all)
                if maps is not None:
                    with self._lock:
                        self._latest, self._received = maps, time.perf_counter()
                    self._ready.set()
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()

    def get(self):
        return self.get_with_timestamp()[:2]

    def get_with_timestamp(self):
        """Return raw/visual maps and their host processing time under the same lock."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"Tactile {self.config.port}: serial reader failed") from self._error
            if self._latest is None or time.perf_counter() - self._received > self.config.timeout_s:
                raise TimeoutError(f"Tactile {self.config.port}: no fresh calibrated frame within {self.config.timeout_s}s")
            return *(arr.copy() for arr in self._latest), self._received

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        with self._lock:
            self._latest = None
