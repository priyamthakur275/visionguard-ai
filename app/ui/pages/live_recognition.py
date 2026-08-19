"""
live_recognition.py
====================
Live webcam recognition page. Pipeline per frame:

    Face Detection -> Embedding Extraction -> Cosine Similarity -> Recognition

Bounding boxes are drawn green for known identities and red for unknown
ones, with name, confidence, distance, and FPS overlaid. A per-track
label smoother prevents flicker as detections jitter frame to frame.
"""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import customtkinter as ctk

from app.core.camera_utils import CameraStream
from app.core.distance_utils import bbox_pixel_width, classify_distance, estimate_distance_cm
from app.core.image_utils import bgr_to_pil
from app.core.recognition_utils import RecognitionEngine
from app.core.recognition_worker import LatestFrameRecognitionWorker
from app.core.tracking_utils import CentroidTracker
from app.logger import get_logger
from app.ui import theme
from app.ui.context import AppContext
from app.ui.widgets.dialogs import show_info_dialog
from app.ui.widgets.distance_indicator import DistanceIndicator

logger = get_logger(__name__)

KNOWN_COLOR_BGR = (46, 204, 113)   # green
UNKNOWN_COLOR_BGR = (60, 60, 231)  # red


class LiveRecognitionPage(ctk.CTkFrame):
    def __init__(self, master, context: AppContext, on_status_change=None):
        super().__init__(master, fg_color=theme.BG_PRIMARY, corner_radius=0)
        self._context = context
        self._on_status_change = on_status_change

        self._camera: CameraStream | None = None
        self._recognition_engine: RecognitionEngine | None = None
        self._recognition_worker: LatestFrameRecognitionWorker | None = None
        self._tracker = CentroidTracker(smoothing_window=context.config.recognition.recognition_smoothing_window)
        self._is_running = False
        self._update_job = None

        self._build_header()
        self._build_video_area()
        self._build_control_bar()

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_LARGE, pady=(theme.PAD_LARGE, theme.PAD_SMALL))
        ctk.CTkLabel(header, text="Live Recognition", font=theme.FONT_TITLE, text_color=theme.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Real-time face detection, recognition, and distance estimation",
            font=theme.FONT_SUBTITLE, text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w")

    def _build_video_area(self) -> None:
        self._video_frame = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        self._video_frame.pack(fill="both", expand=True, padx=theme.PAD_LARGE, pady=theme.PAD_SMALL)

        self._video_label = ctk.CTkLabel(self._video_frame, text="Camera is off. Click 'Start Camera' to begin.", font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY)
        self._video_label.pack(expand=True, padx=theme.PAD_MEDIUM, pady=theme.PAD_MEDIUM)

    def _build_control_bar(self) -> None:
        control_bar = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        control_bar.pack(fill="x", padx=theme.PAD_LARGE, pady=(0, theme.PAD_LARGE))

        left = ctk.CTkFrame(control_bar, fg_color="transparent")
        left.pack(side="left", padx=theme.PAD_MEDIUM, pady=theme.PAD_MEDIUM)

        self._start_button = ctk.CTkButton(
            left, text="▶  Start Camera", width=160, height=38, fg_color=theme.SUCCESS, hover_color="#27AE60",
            font=theme.FONT_BUTTON, command=self.start,
        )
        self._start_button.pack(side="left", padx=4)

        self._stop_button = ctk.CTkButton(
            left, text="■  Stop Camera", width=160, height=38, fg_color=theme.DANGER, hover_color="#C0392B",
            font=theme.FONT_BUTTON, command=self.stop, state="disabled",
        )
        self._stop_button.pack(side="left", padx=4)

        stats = ctk.CTkFrame(control_bar, fg_color="transparent")
        stats.pack(side="left", padx=theme.PAD_LARGE)

        self._fps_label = ctk.CTkLabel(stats, text="FPS: --", font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY)
        self._fps_label.pack(side="left", padx=10)

        self._time_label = ctk.CTkLabel(stats, text="", font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY)
        self._time_label.pack(side="left", padx=10)

        self._distance_indicator = DistanceIndicator(control_bar)
        self._distance_indicator.pack(side="right", padx=theme.PAD_MEDIUM)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._is_running:
            return

        try:
            self._context.face_engine.ensure_loaded()
            self._recognition_engine = RecognitionEngine(
                self._context.config, self._context.face_engine, self._context.database
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize recognition engine: %s", exc)
            show_info_dialog(self, "Model Load Error", f"Could not load the recognition model:\n{exc}", success=False)
            return

        self._camera = CameraStream(self._context.config)
        if not self._camera.start():
            show_info_dialog(self, "Camera Unavailable", "Could not open the configured camera device. Check Settings and close other camera apps.", success=False)
            self._camera = None
            return
        self._recognition_worker = LatestFrameRecognitionWorker(self._recognition_engine)
        self._recognition_worker.start()

        self._is_running = True
        self._start_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        if self._on_status_change:
            self._on_status_change("camera", "Active")
            self._on_status_change("recognition", "Running")
        self._update_frame()

    def stop(self) -> None:
        self._is_running = False
        if self._update_job is not None:
            self.after_cancel(self._update_job)
            self._update_job = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        if self._recognition_worker is not None:
            self._recognition_worker.stop()
            self._recognition_worker = None
        self._video_label.configure(image=None, text="Camera is off. Click 'Start Camera' to begin.")
        self._distance_indicator.clear()
        self._fps_label.configure(text="FPS: --")
        self._start_button.configure(state="normal")
        self._stop_button.configure(state="disabled")
        if self._on_status_change:
            self._on_status_change("camera", "Idle")
            self._on_status_change("recognition", "Idle")

    # ------------------------------------------------------------------
    def _update_frame(self) -> None:
        if not self._is_running or self._camera is None:
            return

        ok, frame = self._camera.read()
        if not ok or frame is None:
            if self._camera.has_failed:
                self.stop()
                show_info_dialog(self, "Camera Unavailable", "Camera stream disconnected or failed.", success=False)
                return
            self._update_job = self.after(15, self._update_frame)
            return

        if self._recognition_worker is not None:
            self._recognition_worker.submit(frame)
            work = self._recognition_worker.latest_result()
            if work is not None:
                if work.error is not None:
                    self.stop()
                    show_info_dialog(self, "Recognition Error", str(work.error), success=False)
                    return
                annotated = self._process_and_annotate(work.frame, work.results)
                self._render_frame(annotated)
                self._fps_label.configure(text=f"Camera: {self._camera.fps:.1f} | Inference: {work.inference_fps:.1f} FPS")

        self._time_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        self._update_job = self.after(15, self._update_frame)

    def _process_and_annotate(self, frame_bgr, results):
        config = self._context.config
        detections = [res.bbox for _, res in results]
        self._tracker.update(detections)

        latest_reading = None
        for face, result in results:
            track_id = self._tracker.find_track_id_for_bbox(result.bbox)
            stable_name = result.name
            if track_id != -1:
                smoother = self._tracker.get_smoother(track_id)
                stable_name = smoother.update(result.name)

            color = KNOWN_COLOR_BGR if result.is_known else UNKNOWN_COLOR_BGR
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            pixel_width = bbox_pixel_width(result.bbox)
            distance_cm = estimate_distance_cm(pixel_width, config)
            reading = classify_distance(distance_cm, config)
            latest_reading = reading

            label_text = f"{stable_name}"
            if result.is_known:
                label_text += f" ({result.similarity * 100:.0f}%)"
            distance_text = f"{distance_cm:.0f}cm" if distance_cm >= 0 else "--"

            self._draw_label(frame_bgr, x1, y1, label_text, distance_text, color)

        if latest_reading is not None:
            self._distance_indicator.update_reading(latest_reading)
        else:
            self._distance_indicator.clear()

        return frame_bgr

    @staticmethod
    def _draw_label(frame_bgr, x1: int, y1: int, name_text: str, distance_text: str, color) -> None:
        label = f"{name_text} | {distance_text}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - text_h - 12)
        cv2.rectangle(frame_bgr, (x1, top), (x1 + text_w + 10, top + text_h + 10), color, -1)
        cv2.putText(
            frame_bgr, label, (x1 + 5, top + text_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA,
        )

    def _render_frame(self, frame_bgr) -> None:
        pil_image = bgr_to_pil(frame_bgr)
        width = max(self._video_frame.winfo_width() - 2 * theme.PAD_MEDIUM, 640)
        height = int(width * pil_image.height / pil_image.width)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(width, height))
        self._video_label.configure(image=ctk_image, text="")
        self._video_label.image = ctk_image

    def on_page_hidden(self) -> None:
        """Called by the main window when navigating away, to release
        the camera device rather than leaving it open in the background."""
        if self._is_running:
            self.stop()
