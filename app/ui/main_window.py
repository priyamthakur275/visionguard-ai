"""
main_window.py
===============
Top-level application window. Wires together the sidebar and page
router, and owns the shared AppContext passed into every page.
"""

from __future__ import annotations

import sys

import customtkinter as ctk

from app.logger import get_logger
from app.ui import theme
from app.ui.context import AppContext
from app.ui.pages.dashboard import DashboardPage
from app.ui.pages.enroll_person import EnrollPersonPage
from app.ui.pages.live_recognition import LiveRecognitionPage
from app.ui.pages.registered_persons import RegisteredPersonsPage
from app.ui.pages.robot_mode import RobotModePage
from app.ui.pages.settings import SettingsPage
from app.ui.sidebar import Sidebar
from app.ui.widgets.dialogs import show_confirm_dialog

logger = get_logger(__name__)


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._context = AppContext.create()
        cfg = self._context.config.app

        self.title(f"{cfg.name} — Smart Face Recognition & Robot Tracking System")
        self.geometry(f"{cfg.window_width}x{cfg.window_height}")
        self.minsize(1024, 650)

        ctk.set_appearance_mode(cfg.theme)
        ctk.set_default_color_theme(cfg.color_theme)
        self.configure(fg_color=theme.BG_PRIMARY)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._pages: dict = {}
        self._current_page_key: str | None = None

        self._sidebar = Sidebar(self, on_navigate=self.show_page, on_exit=self._handle_exit)
        self._sidebar.grid(row=0, column=0, sticky="ns")

        self._page_container = ctk.CTkFrame(self, fg_color=theme.BG_PRIMARY, corner_radius=0)
        self._page_container.grid(row=0, column=1, sticky="nsew")
        self._page_container.grid_columnconfigure(0, weight=1)
        self._page_container.grid_rowconfigure(0, weight=1)

        self._build_pages()
        self.show_page("dashboard")

        self.protocol("WM_DELETE_WINDOW", self._handle_exit)

    # ------------------------------------------------------------------
    def _build_pages(self) -> None:
        self._pages["dashboard"] = DashboardPage(self._page_container, self._context)
        self._pages["enroll"] = EnrollPersonPage(self._page_container, self._context, on_enrolled=self._handle_enrolled)
        self._pages["registered"] = RegisteredPersonsPage(self._page_container, self._context, on_change=self._handle_db_changed)
        self._pages["live"] = LiveRecognitionPage(self._page_container, self._context, on_status_change=self._handle_status_change)
        self._pages["robot"] = RobotModePage(self._page_container, self._context, on_status_change=self._handle_status_change)
        self._pages["settings"] = SettingsPage(self._page_container, self._context, on_saved=self._handle_settings_saved)

        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            logger.warning("Attempted to navigate to unknown page '%s'", key)
            return

        previous_key = self._current_page_key
        if previous_key is not None and previous_key != key:
            previous_page = self._pages[previous_key]
            if hasattr(previous_page, "on_page_hidden"):
                previous_page.on_page_hidden()

        page = self._pages[key]
        if key == "dashboard":
            page.refresh()
        if key == "registered":
            page.refresh()

        page.tkraise()
        self._current_page_key = key

    # ------------------------------------------------------------------
    def _handle_enrolled(self, person_name: str, accepted_count: int) -> None:
        dashboard: DashboardPage = self._pages["dashboard"]
        dashboard.log_activity(f"Enrolled '{person_name}' with {accepted_count} embedding(s)")
        dashboard.refresh()
        self._pages["registered"].refresh()

    def _handle_db_changed(self) -> None:
        dashboard: DashboardPage = self._pages["dashboard"]
        dashboard.refresh()

    def _handle_status_change(self, field: str, value: str) -> None:
        dashboard: DashboardPage = self._pages["dashboard"]
        if field == "camera":
            dashboard.set_camera_status(value)
        elif field == "recognition":
            dashboard.set_recognition_status(value)
        elif field == "robot":
            dashboard.set_robot_status(value)
        dashboard.log_activity(f"{field.capitalize()} status -> {value}")

    def _handle_settings_saved(self) -> None:
        dashboard: DashboardPage = self._pages["dashboard"]
        dashboard.log_activity("Settings updated")

    # ------------------------------------------------------------------
    def _handle_exit(self) -> None:
        def do_exit():
            for page in self._pages.values():
                if hasattr(page, "on_page_hidden"):
                    page.on_page_hidden()
            self.destroy()
            sys.exit(0)

        show_confirm_dialog(
            self, "Exit VisionGuard AI", "Are you sure you want to exit the application?", on_confirm=do_exit,
            confirm_text="Exit",
        )
