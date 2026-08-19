"""Latest-frame recognition worker.

Inference never runs on CustomTkinter's event loop.  The input queue has one
slot: when the camera is faster than inference, stale frames are discarded.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.recognition_utils import RecognitionEngine
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionWorkResult:
    frame: np.ndarray
    results: list
    inference_fps: float
    error: Optional[Exception] = None


class LatestFrameRecognitionWorker:
    def __init__(self, engine: RecognitionEngine):
        self._engine = engine
        self._input: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._output: queue.Queue[RecognitionWorkResult] = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="recognition-worker", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        try:
            self._input.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._input.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input.put_nowait(frame.copy())
            except queue.Full:
                pass

    def latest_result(self) -> Optional[RecognitionWorkResult]:
        latest = None
        while True:
            try:
                latest = self._output.get_nowait()
            except queue.Empty:
                return latest

    def _publish(self, result: RecognitionWorkResult) -> None:
        try:
            self._output.put_nowait(result)
        except queue.Full:
            try:
                self._output.get_nowait()
            except queue.Empty:
                pass
            self._output.put_nowait(result)

    def _run(self) -> None:
        while self._running:
            try:
                frame = self._input.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                results = self._engine.recognize_frame(frame)
                elapsed = max(time.perf_counter() - started, 1e-6)
                self._publish(RecognitionWorkResult(frame, results, 1.0 / elapsed))
            except Exception as exc:  # surfaced to UI on its own thread
                logger.exception("Recognition worker failed")
                self._publish(RecognitionWorkResult(frame, [], 0.0, exc))

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
