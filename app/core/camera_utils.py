"""
camera_utils.py
================
Threaded webcam capture wrapper. Reading frames on a background thread
keeps the CustomTkinter UI thread free to redraw and stay responsive,
while always exposing the most recently captured frame to consumers.

Handles the "camera unavailable" failure mode gracefully: if the device
cannot be opened, `CameraStream.start()` returns False instead of
raising, and the UI displays a friendly error state rather than crashing.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

from app.config import AppConfig
from app.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None


class CameraStream:
    """Background-threaded VideoCapture wrapper with FPS tracking."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._capture: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = False
        self._consecutive_failures = 0

        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer_start = time.time()

    def start(self) -> bool:
        """Attempt to open the configured camera device and begin
        capturing frames on a background thread. Returns True on
        success, False if the camera could not be opened."""
        if cv2 is None:
            logger.error("OpenCV is not installed; cannot open camera.")
            return False

        device_index = self._config.camera.device_index
        capture = cv2.VideoCapture(device_index)
        if not capture.isOpened():
            logger.error("Unable to open camera at index %d", device_index)
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera.frame_height)
        capture.set(cv2.CAP_PROP_FPS, self._config.camera.target_fps)

        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera stream started on device index %d", device_index)
        return True

    def _capture_loop(self) -> None:
        while self._running and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._consecutive_failures += 1
                time.sleep(0.05)
                continue

            self._consecutive_failures = 0

            with self._lock:
                self._latest_frame = frame
                self._frame_ready = True

            self._frame_count += 1
            elapsed = time.time() - self._fps_timer_start
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer_start = time.time()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (frame_available, frame_copy) for the most recent frame."""
        with self._lock:
            if not self._frame_ready or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_failed(self) -> bool:
        """True only after a sustained capture failure, not normal startup latency."""
        return self._consecutive_failures >= 10

    def stop(self) -> None:
        """Stop the capture thread and release the camera device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        logger.info("Camera stream stopped.")
