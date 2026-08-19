"""
enroll_person.py
=================
Enrollment page: capture a person's name, let the user pick 2-20 image
files via the native file dialog, then run each image through the
detect-single-face -> align -> embed -> normalize pipeline, persisting
every accepted embedding individually (never averaged).
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk

from app.core.embedding_utils import l2_normalize
from app.core.file_utils import copy_image_into_gallery, is_supported_image
from app.core.image_utils import ImageLoadError, load_image_bgr
from app.logger import get_logger
from app.ui import theme
from app.ui.context import AppContext
from app.ui.widgets.dialogs import DuplicateIdentityDialog, ProgressDialog, show_info_dialog

logger = get_logger(__name__)

_REJECTION_MESSAGES = {
    "no_face": "No face detected",
    "multiple_faces": "Multiple faces detected",
    "corrupted": "File is corrupted or unreadable",
    "unsupported": "Unsupported file type",
}


class EnrollPersonPage(ctk.CTkFrame):
    def __init__(self, master, context: AppContext, on_enrolled=None):
        super().__init__(master, fg_color=theme.BG_PRIMARY, corner_radius=0)
        self._context = context
        self._on_enrolled = on_enrolled
        self._selected_files: List[Path] = []
        self._consent_var = ctk.BooleanVar(value=False)

        self._build_header()
        self._build_form()

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_LARGE, pady=(theme.PAD_LARGE, theme.PAD_MEDIUM))
        ctk.CTkLabel(header, text="Enroll Person", font=theme.FONT_TITLE, text_color=theme.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Register a new identity by uploading 2-20 clear face images",
            font=theme.FONT_SUBTITLE,
            text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w")

    def _build_form(self) -> None:
        card = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        card.pack(fill="both", expand=True, padx=theme.PAD_LARGE, pady=(0, theme.PAD_LARGE))

        ctk.CTkLabel(card, text="Person's Name", font=theme.FONT_BODY, text_color=theme.TEXT_PRIMARY).pack(
            anchor="w", padx=theme.PAD_LARGE, pady=(theme.PAD_LARGE, 4)
        )
        self._name_entry = ctk.CTkEntry(
            card, placeholder_text="e.g. John Doe", width=360, height=38,
            fg_color=theme.BG_SECONDARY, border_color=theme.BORDER,
        )
        self._name_entry.pack(anchor="w", padx=theme.PAD_LARGE)

        upload_row = ctk.CTkFrame(card, fg_color="transparent")
        upload_row.pack(anchor="w", padx=theme.PAD_LARGE, pady=theme.PAD_MEDIUM)

        self._upload_button = ctk.CTkButton(
            upload_row, text="📁  Upload Images", width=180, height=38,
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER, font=theme.FONT_BUTTON,
            command=self._handle_upload_click,
        )
        self._upload_button.pack(side="left")

        self._selection_label = ctk.CTkLabel(
            upload_row, text="No images selected", font=theme.FONT_CARD_LABEL, text_color=theme.TEXT_SECONDARY
        )
        self._selection_label.pack(side="left", padx=theme.PAD_MEDIUM)

        cfg = self._context.config.enrollment
        ctk.CTkLabel(
            card,
            text=(
                f"Minimum {cfg.min_images} images required · "
                f"Recommended {cfg.recommended_images}-{cfg.max_images} images"
            ),
            font=theme.FONT_CARD_LABEL,
            text_color=theme.TEXT_MUTED,
        ).pack(anchor="w", padx=theme.PAD_LARGE)

        self._thumbnails_frame = ctk.CTkScrollableFrame(
            card, fg_color=theme.BG_SECONDARY, height=180, corner_radius=8
        )
        self._thumbnails_frame.pack(fill="x", padx=theme.PAD_LARGE, pady=theme.PAD_MEDIUM)

        ctk.CTkCheckBox(
            card, text="I have consent to enroll and store this person's biometric data locally.",
            variable=self._consent_var, font=theme.FONT_CARD_LABEL, text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w", padx=theme.PAD_LARGE, pady=(0, theme.PAD_SMALL))

        self._enroll_button = ctk.CTkButton(
            card, text="✅  Enroll Person", width=200, height=42,
            fg_color=theme.SUCCESS, hover_color="#27AE60", font=theme.FONT_BUTTON,
            command=self._handle_enroll_click,
        )
        self._enroll_button.pack(anchor="w", padx=theme.PAD_LARGE, pady=(theme.PAD_SMALL, theme.PAD_LARGE))

    # ------------------------------------------------------------------
    def _handle_upload_click(self) -> None:
        max_images = self._context.config.enrollment.max_images
        paths = filedialog.askopenfilenames(
            title="Select Face Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not paths:
            return

        candidates = [Path(p) for p in paths if is_supported_image(Path(p))]
        if len(candidates) > max_images:
            show_info_dialog(
                self,
                "Too Many Images",
                f"Please select at most {max_images} images. You selected {len(candidates)}.",
                success=False,
            )
            candidates = candidates[:max_images]

        self._selected_files = candidates
        self._selection_label.configure(text=f"{len(self._selected_files)} image(s) selected")
        self._render_thumbnails()

    def _render_thumbnails(self) -> None:
        for widget in self._thumbnails_frame.winfo_children():
            widget.destroy()

        from app.core.image_utils import thumbnail_from_path
        from PIL import Image as PILImage

        row_frame = None
        for idx, path in enumerate(self._selected_files):
            if idx % 6 == 0:
                row_frame = ctk.CTkFrame(self._thumbnails_frame, fg_color="transparent")
                row_frame.pack(anchor="w", fill="x")
            try:
                pil_img = thumbnail_from_path(path, size=(80, 80))
            except Exception:  # noqa: BLE001
                pil_img = PILImage.new("RGB", (80, 80), color=(60, 60, 60))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(80, 80))
            label = ctk.CTkLabel(row_frame, image=ctk_img, text="")
            label.image = ctk_img  # keep reference
            label.pack(side="left", padx=6, pady=6)

    # ------------------------------------------------------------------
    def _handle_enroll_click(self) -> None:
        person_name = self._name_entry.get().strip()
        min_images = self._context.config.enrollment.min_images

        if not person_name:
            show_info_dialog(self, "Missing Name", "Please enter the person's name.", success=False)
            return
        if not self._consent_var.get():
            show_info_dialog(self, "Consent Required", "Confirm that you have consent before enrolling biometric data.", success=False)
            return
        if len(person_name) > 80 or not any(char.isalnum() for char in person_name):
            show_info_dialog(self, "Invalid Name", "Use a meaningful name up to 80 characters.", success=False)
            return
        if len(self._selected_files) < min_images:
            show_info_dialog(
                self, "Not Enough Images",
                f"Please select at least {min_images} images. You selected {len(self._selected_files)}.",
                success=False,
            )
            return

        if self._context.database.person_exists(person_name):
            DuplicateIdentityDialog(self, person_name, on_choice=lambda choice: self._on_duplicate_choice(choice, person_name))
        else:
            self._run_enrollment(person_name, overwrite=True)

    def _on_duplicate_choice(self, choice: Optional[str], person_name: str) -> None:
        if choice is None:
            return  # Cancel
        self._run_enrollment(person_name, overwrite=(choice == "overwrite"))

    # ------------------------------------------------------------------
    def _run_enrollment(self, person_name: str, overwrite: bool) -> None:
        self._enroll_button.configure(state="disabled")
        self._upload_button.configure(state="disabled")
        progress = ProgressDialog(self, title="Enrolling Person")

        thread = threading.Thread(
            target=self._process_enrollment_worker,
            args=(person_name, overwrite, progress),
            daemon=True,
        )
        thread.start()

    def _process_enrollment_worker(self, person_name: str, overwrite: bool, progress: ProgressDialog) -> None:
        config = self._context.config
        face_engine = self._context.face_engine
        database = self._context.database

        accepted_embeddings = []
        identity_id = database.identity_id_for(person_name) or str(uuid.uuid4())
        representative_image_rel: Optional[str] = None
        accepted_count = 0
        rejected_count = 0
        rejection_details: List[str] = []
        total = len(self._selected_files)

        for idx, source_path in enumerate(self._selected_files, start=1):
            fraction = idx / max(total, 1)
            self.after(0, progress.update_progress, fraction, f"Processing image {idx} of {total}", source_path.name)

            try:
                image_bgr = load_image_bgr(source_path)
            except ImageLoadError:
                rejected_count += 1
                rejection_details.append(f"{source_path.name}: {_REJECTION_MESSAGES['corrupted']}")
                continue

            face, reason = face_engine.detect_single_face(image_bgr)
            if face is None:
                rejected_count += 1
                rejection_details.append(f"{source_path.name}: {_REJECTION_MESSAGES.get(reason, reason)}")
                continue

            normalized_embedding = l2_normalize(face.embedding)
            accepted_embeddings.append(normalized_embedding)
            accepted_count += 1

            stored_path = copy_image_into_gallery(source_path, config.images_root, identity_id)
            if representative_image_rel is None:
                representative_image_rel = str(stored_path.relative_to(config.images_root))

        self.after(0, progress.update_progress, 1.0, "Saving to database...", "")

        if accepted_count > 0:
            database.add_embeddings(
                person_name=person_name,
                embeddings=accepted_embeddings,
                image_paths_added=accepted_count,
                representative_image=representative_image_rel,
                overwrite=overwrite,
                identity_id=identity_id,
            )

        self.after(0, self._finish_enrollment, progress, person_name, accepted_count, rejected_count, rejection_details)

    def _finish_enrollment(
        self, progress: ProgressDialog, person_name: str, accepted: int, rejected: int, rejection_details: List[str]
    ) -> None:
        progress.close()
        self._enroll_button.configure(state="normal")
        self._upload_button.configure(state="normal")

        if accepted == 0:
            show_info_dialog(
                self, "Enrollment Failed",
                "No valid faces were found in the selected images:\n" + "\n".join(rejection_details[:5]),
                success=False,
            )
            return

        summary = f"Enrolled '{person_name}' with {accepted} embedding(s)."
        if rejected:
            summary += f"\n{rejected} image(s) were rejected:\n" + "\n".join(rejection_details[:5])

        show_info_dialog(self, "Enrollment Complete", summary, success=True)

        self._name_entry.delete(0, "end")
        self._consent_var.set(False)
        self._selected_files = []
        self._selection_label.configure(text="No images selected")
        for widget in self._thumbnails_frame.winfo_children():
            widget.destroy()

        if self._on_enrolled:
            self._on_enrolled(person_name, accepted)
