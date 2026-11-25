"""Control a USB GoPro over HTTP and receive RTSP over TCP as RGB frames."""

from __future__ import annotations

import errno
import json
import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _av_available, require_package

from ..opencv import OpenCVCamera, OpenCVCameraConfig
from .configuration_gopro import GoProCameraConfig

if TYPE_CHECKING or _av_available:
    import av

logger = logging.getLogger(__name__)


class GoProCamera(OpenCVCamera):
    def __init__(self, config: GoProCameraConfig):
        super().__init__(
            OpenCVCameraConfig(
                index_or_path=0,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
        )
        self.gopro_config = config
        self._http = build_opener(ProxyHandler({}))
        self._stream_started = False
        self._container: av.container.InputContainer | None = None
        self._frames: Iterator[av.VideoFrame] | None = None

    def __str__(self) -> str:
        return f"GoProCamera({self.gopro_config.ip_address}:{self.gopro_config.port})"

    @property
    def is_connected(self) -> bool:
        return self._container is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # Explicit USB IP selection; never enumerate unrelated /dev/video devices.
        return []

    def _command(self, endpoint: str, **params: int | str) -> dict[str, Any]:
        url = f"http://{self.gopro_config.ip_address}:8080/gopro/{endpoint}"
        if params:
            url += "?" + urlencode(params)
        try:
            with self._http.open(url, timeout=5) as response:
                result = json.load(response)
        except (TimeoutError, URLError) as error:
            if isinstance(error, URLError) and not isinstance(error.reason, TimeoutError):
                raise
            raise TimeoutError(
                f"GoPro HTTP {endpoint} timed out after 5s ({self.gopro_config.ip_address}:8080). "
                "Power-cycle the camera if repeated."
            ) from error
        # HERO13 can report undocumented status=4. Preserve it, but always check error.
        if not isinstance(result, dict) or (endpoint.startswith("webcam/") and result.get("error") != 0):
            raise RuntimeError(
                f"GoPro {endpoint} failed: {result}. "
                "For error=7 (unavailable), check the camera's USB/lens mode."
            )
        return result

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self._command("camera/state")["status"]
            if status.get("8") == 0 and status.get("10") == 0:  # BUSY, ENCODING
                return
            time.sleep(0.1)
        raise TimeoutError("GoPro is busy or recording; webcam was not started.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        require_package("av", extra="av-dep")
        # Also applies when the recorder restores FFmpeg's native stderr callback.
        # Silence native FFmpeg output; Python exceptions still propagate.
        av.logging.set_libav_level(-8)  # AV_LOG_QUIET
        try:
            self._wait_ready()
            self._command("camera/control/wired_usb", p=0)
            # Mark before sending so a timed-out start is also stopped during cleanup.
            self._stream_started = True
            # HERO13's default USB stream is native 1920x1080. Keep frames unchanged.
            self._command("webcam/start", protocol="RTSP")
            url = f"rtsp://{self.gopro_config.ip_address}:{self.gopro_config.port}/live"
            deadline = time.monotonic() + 10
            while True:
                try:
                    # HTTP start returns before the RTSP listener/path is necessarily ready.
                    # Keep webcam running while retrying, or each attempt restarts that startup.
                    self._container = av.open(url, options={"rtsp_transport": "tcp"}, timeout=(10.0, 1.0))
                    break
                except av.error.FFmpegError as error:
                    starting = error.errno == errno.ECONNREFUSED or isinstance(
                        error, av.error.HTTPNotFoundError
                    )
                    if not starting:
                        raise ConnectionError(f"Cannot receive {url} over TCP: {error}") from error
                    if time.monotonic() >= deadline:
                        raise ConnectionError(
                            f"RTSP TCP :{self.gopro_config.port} not ready after 10s. "
                            f"Restart GoPro and check RTSP support. {url}: {error}"
                        ) from error
                    time.sleep(0.25)
            # One decoder thread avoids frame-threading latency.
            self._container.streams.video[0].codec_context.thread_count = 1
            self._frames = self._container.decode(video=0)
            self._start_read_thread()
            if warmup:
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    self.async_read(timeout_ms=10000)
        except BaseException:
            self.disconnect()
            raise

    def _read_from_hardware(self) -> NDArray[Any]:
        if self._frames is None:
            raise DeviceNotConnectedError(f"{self} RTSP decoder is not initialized")
        try:
            return next(self._frames).to_ndarray(format="bgr24")
        except StopIteration as error:
            raise DeviceNotConnectedError(f"{self} RTSP stream ended") from error

    def _cleanup_resources(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                # Closing a PyAV container while another thread decodes it is unsafe.
                raise TimeoutError(f"{self} RTSP reader has not stopped; retaining decoder for cleanup.")
        self._stop_read_thread()
        self._frames = None
        container, self._container = self._container, None
        if container is not None:
            container.close()

    def disconnect(self) -> None:
        try:
            self._cleanup_resources()
        finally:
            if self._stream_started:
                self._stream_started = False
                for endpoint in ("webcam/stop", "webcam/exit"):
                    try:
                        self._command(endpoint)
                    except Exception:
                        logger.exception("Failed to send GoPro %s", endpoint)
