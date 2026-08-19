"""
robot_mode.py
=============
Extends Live Recognition with autonomous tracking: the largest/primary
detected face is treated as the robot's target. Its horizontal offset
from frame-center and estimated distance are converted into a discrete
movement command (LEFT / RIGHT / FORWARD / BACKWARD / STOP), which is
either displayed on-screen (simulate_hardware=true) or transmitted over
serial to real hardware.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import customtkinter as ctk

from app.core.camera_utils import CameraStream
from app.core.distance_utils import bbox_pixel_width, classify_distance, estimate_distance_cm
from app.core.image_utils import bgr_to_pil
from app.core.recognition_utils import RecognitionEngine
from app.core.recognition_worker import LatestFrameRecognitionWorker
from app.core.robot_utils import RobotCommand, decide_command, select_recognized_target
from app.core.tracking_utils import CentroidTracker
from app.logger import get_logger
from app.ui import theme
from app.ui.context import AppContext
from app.ui.widgets.dialogs import show_info_dialog
from app.ui.widgets.distance_indicator import DistanceIndicator

logger = get_logger(__name__)

KNOWN_COLOR_BGR = (46, 204, 113)
UNKNOWN_COLOR_BGR = (60, 60, 231)
TARGET_COLOR_BGR = (255, 191, 0)  # highlight color for the actively-tracked target

_COMMAND_ICONS = {
    RobotCommand.LEFT: "⬅️",
    RobotCommand.RIGHT: "➡️",
    RobotCommand.FORWARD: "⬆️",
    RobotCommand.BACKWARD: "⬇️",
    RobotCommand.STOP: "⏹️",
}


class RobotModePage(ctk.CTkFrame):
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
        self._selected_target = ctk.StringVar(value="Select a recognized person")

        self._build_header()
        self._build_video_area()
        self._build_control_bar()
        self._build_command_panel()

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_LARGE, pady=(theme.PAD_LARGE, theme.PAD_SMALL))
        ctk.CTkLabel(header, text="Robot Mode", font=theme.FONT_TITLE, text_color=theme.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Autonomous person-following with live robot command generation",
            font=theme.FONT_SUBTITLE, text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w")

    def _build_video_area(self) -> None:
        self._video_frame = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        self._video_frame.pack(fill="both", expand=True, padx=theme.PAD_LARGE, pady=theme.PAD_SMALL)
        self._video_label = ctk.CTkLabel(
            self._video_frame, text="Camera is off. Click 'Start Robot Mode' to begin.",
            font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY,
        )
        self._video_label.pack(expand=True, padx=theme.PAD_MEDIUM, pady=theme.PAD_MEDIUM)

    def _build_control_bar(self) -> None:
        control_bar = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        control_bar.pack(fill="x", padx=theme.PAD_LARGE, pady=(0, theme.PAD_SMALL))

        left = ctk.CTkFrame(control_bar, fg_color="transparent")
        left.pack(side="left", padx=theme.PAD_MEDIUM, pady=theme.PAD_MEDIUM)

        self._start_button = ctk.CTkButton(
            left, text="▶  Start Robot Mode", width=180, height=38, fg_color=theme.SUCCESS, hover_color="#27AE60",
            font=theme.FONT_BUTTON, command=self.start,
        )
        self._start_button.pack(side="left", padx=4)

        self._stop_button = ctk.CTkButton(
            left, text="■  Stop", width=120, height=38, fg_color=theme.DANGER, hover_color="#C0392B",
            font=theme.FONT_BUTTON, command=self.stop, state="disabled",
        )
        self._stop_button.pack(side="left", padx=4)

        self._emergency_button = ctk.CTkButton(
            left, text="EMERGENCY STOP", width=165, height=38, fg_color="#8E1720", hover_color="#6F1017",
            font=theme.FONT_BUTTON, command=self.emergency_stop,
        )
        self._emergency_button.pack(side="left", padx=4)

        self._target_menu = ctk.CTkOptionMenu(
            control_bar, values=["Select a recognized person"], variable=self._selected_target,
            fg_color=theme.BG_SECONDARY, button_color=theme.ACCENT,
        )
        self._target_menu.pack(side="left", padx=theme.PAD_MEDIUM)

        self._mode_label = ctk.CTkLabel(
            control_bar,
            text="Mode: SIMULATED" if context_simulated(self._context) else "Mode: HARDWARE (Serial)",
            font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY,
        )
        self._mode_label.pack(side="left", padx=theme.PAD_LARGE)

        self._distance_indicator = DistanceIndicator(control_bar)
        self._distance_indicator.pack(side="right", padx=theme.PAD_MEDIUM)

    def _build_command_panel(self) -> None:
        panel = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        panel.pack(fill="x", padx=theme.PAD_LARGE, pady=(0, theme.PAD_LARGE))

        ctk.CTkLabel(panel, text="Robot Command", font=theme.FONT_HEADING, text_color=theme.TEXT_PRIMARY).pack(
            anchor="w", padx=theme.PAD_MEDIUM, pady=(theme.PAD_MEDIUM, 4)
        )

        self._command_label = ctk.CTkLabel(
            panel, text="⏹️  STOP", font=("Segoe UI", 28, "bold"), text_color=theme.NEUTRAL
        )
        self._command_label.pack(anchor="w", padx=theme.PAD_MEDIUM)

        self._reason_label = ctk.CTkLabel(
            panel, text="Robot idle.", font=theme.FONT_CARD_LABEL, text_color=theme.TEXT_SECONDARY
        )
        self._reason_label.pack(anchor="w", padx=theme.PAD_MEDIUM, pady=(0, theme.PAD_MEDIUM))

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._is_running:
            return

        people = sorted(person.name for person in self._context.database.list_persons())
        if not people:
            show_info_dialog(self, "No Target Available", "Enroll a person before using Robot Mode.", success=False)
            return
        self._target_menu.configure(values=people)
        if self._selected_target.get() not in people:
            self._selected_target.set(people[0])

        try:
            self._context.face_engine.ensure_loaded()
            self._recognition_engine = RecognitionEngine(
                self._context.config, self._context.face_engine, self._context.database
            )
        except Exception as exc:  # noqa: BLE001
            show_info_dialog(self, "Model Load Error", f"Could not load the recognition model:\n{exc}", success=False)
            return

        self._camera = CameraStream(self._context.config)
        if not self._camera.start():
            show_info_dialog(self, "Camera Unavailable", "Could not open the configured camera device.", success=False)
            self._camera = None
            return
        self._recognition_worker = LatestFrameRecognitionWorker(self._recognition_engine)
        self._recognition_worker.start()

        if not self._context.robot_controller.connect():
            show_info_dialog(
                self, "Robot Connection Failed",
                "Could not connect to the robot over serial. Check the Serial Port in Settings, "
                "or enable Simulate Hardware.",
                success=False,
            )
            self._camera.stop()
            self._camera = None
            return

        self._is_running = True
        self._start_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        if self._on_status_change:
            self._on_status_change("camera", "Active")
            self._on_status_change("robot", "Simulated" if self._context.config.robot.simulate_hardware else "Connected")
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
        self._context.robot_controller.disconnect()
        self._video_label.configure(image=None, text="Camera is off. Click 'Start Robot Mode' to begin.")
        self._distance_indicator.clear()
        self._command_label.configure(text="⏹️  STOP", text_color=theme.NEUTRAL)
        self._reason_label.configure(text="Robot idle.")
        self._start_button.configure(state="normal")
        self._stop_button.configure(state="disabled")
        if self._on_status_change:
            self._on_status_change("camera", "Idle")
            self._on_status_change("robot", "Simulated" if self._context.config.robot.simulate_hardware else "Disconnected")

    # ------------------------------------------------------------------
    def _update_frame(self) -> None:
        if not self._is_running or self._camera is None:
            return

        ok, frame = self._camera.read()
        if not ok or frame is None:
            if self._camera.has_failed:
                self.emergency_stop("Camera input failed; robot stopped.")
                return
            self._update_job = self.after(15, self._update_frame)
            return
        if self._recognition_worker is not None:
            self._recognition_worker.submit(frame)
            work = self._recognition_worker.latest_result()
            if work is not None:
                if work.error is not None:
                    self.emergency_stop("Recognition failed; robot stopped.")
                    show_info_dialog(self, "Recognition Error", str(work.error), success=False)
                    return
                annotated = self._process_frame(work.frame, work.results)
                self._render_frame(annotated)

        self._update_job = self.after(15, self._update_frame)

    def _process_frame(self, frame_bgr, results):
        config = self._context.config
        detections = [res.bbox for _, res in results]
        active_tracks = self._tracker.update(detections)

        # Draw every detected face.
        for face, result in results:
            color = KNOWN_COLOR_BGR if result.is_known else UNKNOWN_COLOR_BGR
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = result.name if not result.is_known else f"{result.name} ({result.similarity * 100:.0f}%)"
            cv2.putText(frame_bgr, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        target_name = self._selected_target.get()
        target_face = select_recognized_target(results, target_name)
        if target_face is None:
            self._distance_indicator.clear()
            self._update_command_display(RobotCommand.STOP, f"Target '{target_name}' not visible.")
            self._context.robot_controller.send_command(RobotCommand.STOP)
            return frame_bgr

        # A known, explicitly selected identity is the only follow target.
        target_bbox = target_face.bbox
        x1, y1, x2, y2 = target_bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), TARGET_COLOR_BGR, 3)

        pixel_width = bbox_pixel_width(target_bbox)
        distance_cm = estimate_distance_cm(pixel_width, config)
        reading = classify_distance(distance_cm, config)
        self._distance_indicator.update_reading(reading)

        face_center_x = (x1 + x2) / 2.0
        frame_width = frame_bgr.shape[1]
        decision = decide_command(frame_width, face_center_x, distance_cm, config)
        self._update_command_display(decision.command, decision.reason)
        self._context.robot_controller.send_command(decision.command)

        return frame_bgr

    def _update_command_display(self, command: RobotCommand, reason: str) -> None:
        icon = _COMMAND_ICONS.get(command, "⏹️")
        color = {
            RobotCommand.LEFT: theme.ACCENT,
            RobotCommand.RIGHT: theme.ACCENT,
            RobotCommand.FORWARD: theme.SUCCESS,
            RobotCommand.BACKWARD: theme.WARNING,
            RobotCommand.STOP: theme.NEUTRAL,
        }.get(command, theme.NEUTRAL)
        self._command_label.configure(text=f"{icon}  {command.value}", text_color=color)
        self._reason_label.configure(text=reason)

    def emergency_stop(self, reason: str = "Emergency stop activated.") -> None:
        """Immediately command STOP, then end the session and release hardware."""
        self._context.robot_controller.send_command(RobotCommand.STOP)
        self._update_command_display(RobotCommand.STOP, reason)
        if self._is_running:
            self.stop()

    def _render_frame(self, frame_bgr) -> None:
        pil_image = bgr_to_pil(frame_bgr)
        width = max(self._video_frame.winfo_width() - 2 * theme.PAD_MEDIUM, 640)
        height = int(width * pil_image.height / pil_image.width)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(width, height))
        self._video_label.configure(image=ctk_image, text="")
        self._video_label.image = ctk_image

    def on_page_hidden(self) -> None:
        if self._is_running:
            self.stop()


def context_simulated(context: AppContext) -> bool:
    return context.config.robot.simulate_hardware
