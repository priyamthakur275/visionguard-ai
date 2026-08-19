"""
registered_persons.py
======================
Searchable table of all enrolled identities with View / Update / Delete
actions per row. CustomTkinter has no native table widget, so rows are
rendered as a grid of labels/buttons inside a scrollable frame - a
common, well-supported pattern for CTk-based desktop apps.
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk

from app.core.database_utils import PersonMetadata
from app.core.embedding_utils import l2_normalize
from app.core.file_utils import copy_image_into_gallery, delete_person_directory, is_supported_image
from app.core.image_utils import ImageLoadError, load_image_bgr, thumbnail_from_path
from app.ui import theme
from app.ui.context import AppContext
from app.ui.widgets.dialogs import ProgressDialog, show_confirm_dialog, show_info_dialog


class RegisteredPersonsPage(ctk.CTkFrame):
    def __init__(self, master, context: AppContext, on_change=None):
        super().__init__(master, fg_color=theme.BG_PRIMARY, corner_radius=0)
        self._context = context
        self._on_change = on_change
        self._search_query = ""

        self._build_header()
        self._build_search_bar()
        self._build_table_container()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_LARGE, pady=(theme.PAD_LARGE, theme.PAD_SMALL))
        ctk.CTkLabel(header, text="Registered Persons", font=theme.FONT_TITLE, text_color=theme.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(
            header, text="View, update, or remove enrolled identities",
            font=theme.FONT_SUBTITLE, text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w")

    def _build_search_bar(self) -> None:
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=theme.PAD_LARGE, pady=(0, theme.PAD_SMALL))

        self._search_entry = ctk.CTkEntry(
            search_frame, placeholder_text="🔎 Search by name...", width=320, height=36,
            fg_color=theme.BG_CARD, border_color=theme.BORDER,
        )
        self._search_entry.pack(side="left")
        self._search_entry.bind("<KeyRelease>", self._on_search_change)

        ctk.CTkButton(
            search_frame, text="↻ Refresh", width=100, height=36, fg_color=theme.NEUTRAL, hover_color="#4A5764",
            command=self.refresh,
        ).pack(side="left", padx=theme.PAD_SMALL)

    def _build_table_container(self) -> None:
        self._table_frame = ctk.CTkScrollableFrame(self, fg_color=theme.BG_CARD, corner_radius=theme.CORNER_RADIUS)
        self._table_frame.pack(fill="both", expand=True, padx=theme.PAD_LARGE, pady=(0, theme.PAD_LARGE))
        for col, weight in zip(range(6), (0, 2, 2, 1, 1, 3)):
            self._table_frame.grid_columnconfigure(col, weight=weight)

    # ------------------------------------------------------------------
    def _on_search_change(self, _event=None) -> None:
        self._search_query = self._search_entry.get().strip().lower()
        self.refresh()

    def refresh(self) -> None:
        for widget in self._table_frame.winfo_children():
            widget.destroy()

        headers = ["Photo", "Name", "Enrollment Date", "Images", "Embeddings", "Actions"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(
                self._table_frame, text=text, font=theme.FONT_BUTTON, text_color=theme.TEXT_SECONDARY
            ).grid(row=0, column=col, sticky="w", padx=10, pady=(10, 6))

        persons: List[PersonMetadata] = sorted(self._context.database.list_persons(), key=lambda p: p.name.lower())
        if self._search_query:
            persons = [p for p in persons if self._search_query in p.name.lower()]

        if not persons:
            ctk.CTkLabel(
                self._table_frame, text="No registered persons found.", font=theme.FONT_BODY,
                text_color=theme.TEXT_MUTED,
            ).grid(row=1, column=0, columnspan=6, sticky="w", padx=10, pady=20)
            return

        for row_idx, person in enumerate(persons, start=1):
            self._render_row(row_idx, person)

    def _render_row(self, row_idx: int, person: PersonMetadata) -> None:
        config = self._context.config
        thumb_path = None
        if person.representative_image:
            thumb_path = config.images_root / person.representative_image

        pil_thumb = thumbnail_from_path(thumb_path, size=(48, 48))
        ctk_thumb = ctk.CTkImage(light_image=pil_thumb, dark_image=pil_thumb, size=(48, 48))
        photo_label = ctk.CTkLabel(self._table_frame, image=ctk_thumb, text="")
        photo_label.image = ctk_thumb
        photo_label.grid(row=row_idx, column=0, padx=10, pady=8)

        ctk.CTkLabel(self._table_frame, text=person.name, font=theme.FONT_BODY, text_color=theme.TEXT_PRIMARY).grid(
            row=row_idx, column=1, sticky="w", padx=10
        )
        ctk.CTkLabel(
            self._table_frame, text=person.enrollment_date.split("T")[0], font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY
        ).grid(row=row_idx, column=2, sticky="w", padx=10)
        ctk.CTkLabel(
            self._table_frame, text=str(person.image_count), font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY
        ).grid(row=row_idx, column=3, sticky="w", padx=10)
        ctk.CTkLabel(
            self._table_frame, text=str(person.embedding_count), font=theme.FONT_BODY, text_color=theme.TEXT_SECONDARY
        ).grid(row=row_idx, column=4, sticky="w", padx=10)

        actions = ctk.CTkFrame(self._table_frame, fg_color="transparent")
        actions.grid(row=row_idx, column=5, sticky="w", padx=6)

        ctk.CTkButton(
            actions, text="View", width=64, height=28, fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
            command=lambda p=person: self._view_person(p),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            actions, text="Update", width=70, height=28, fg_color=theme.WARNING, hover_color="#D4AC0D",
            text_color="#1A1D23",
            command=lambda p=person: self._update_person(p),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            actions, text="Delete", width=64, height=28, fg_color=theme.DANGER, hover_color="#C0392B",
            command=lambda p=person: self._delete_person(p),
        ).pack(side="left", padx=3)

    # ------------------------------------------------------------------
    def _view_person(self, person: PersonMetadata) -> None:
        config = self._context.config
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Person Details - {person.name}")
        dialog.geometry("420x480")
        dialog.configure(fg_color=theme.BG_SECONDARY)
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text=person.name, font=theme.FONT_HEADING, text_color=theme.TEXT_PRIMARY).pack(pady=(20, 6))
        ctk.CTkLabel(
            dialog,
            text=(
                f"Enrolled: {person.enrollment_date}\n"
                f"Last updated: {person.last_updated}\n"
                f"Images: {person.image_count}   Embeddings: {person.embedding_count}"
            ),
            font=theme.FONT_CARD_LABEL, text_color=theme.TEXT_SECONDARY, justify="center",
        ).pack(pady=(0, 12))

        gallery = ctk.CTkScrollableFrame(dialog, fg_color=theme.BG_CARD, width=380, height=320)
        gallery.pack(padx=20, pady=10, fill="both", expand=True)

        from app.core.file_utils import list_person_images
        image_paths = list_person_images(config.images_root, person.identity_id)

        row_frame = None
        for idx, path in enumerate(image_paths):
            if idx % 4 == 0:
                row_frame = ctk.CTkFrame(gallery, fg_color="transparent")
                row_frame.pack(anchor="w")
            pil_thumb = thumbnail_from_path(path, size=(80, 80))
            ctk_thumb = ctk.CTkImage(light_image=pil_thumb, dark_image=pil_thumb, size=(80, 80))
            lbl = ctk.CTkLabel(row_frame, image=ctk_thumb, text="")
            lbl.image = ctk_thumb
            lbl.pack(side="left", padx=6, pady=6)

        ctk.CTkButton(
            dialog, text="Close", width=100, fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
            command=dialog.destroy,
        ).pack(pady=(0, 16))

    def _update_person(self, person: PersonMetadata) -> None:
        """Allow the user to append additional images to an existing
        identity using the same detect -> embed -> normalize pipeline."""
        maximum = self._context.config.enrollment.max_images
        remaining = maximum - person.image_count
        if remaining <= 0:
            show_info_dialog(self, "Image Limit Reached", f"{person.name} already has {person.image_count}/{maximum} images.", success=False)
            return
        paths = filedialog.askopenfilenames(
            title=f"Add More Images for {person.name}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not paths:
            return

        candidate_paths = [p for p in paths if is_supported_image(Path(p))]
        if not candidate_paths:
            show_info_dialog(self, "No Valid Images", "None of the selected files are supported image types.", success=False)
            return
        if len(candidate_paths) > remaining:
            show_info_dialog(self, "Image Limit", f"Only {remaining} additional image(s) can be added.", success=False)
            return

        progress = ProgressDialog(self, title=f"Updating {person.name}")

        import threading
        thread = threading.Thread(target=self._process_update_worker, args=(person, candidate_paths, progress), daemon=True)
        thread.start()

    def _process_update_worker(self, person: PersonMetadata, paths, progress: ProgressDialog) -> None:
        config = self._context.config
        face_engine = self._context.face_engine
        database = self._context.database

        accepted_embeddings = []
        accepted = 0
        rejected = 0
        total = len(paths)

        for idx, raw_path in enumerate(paths, start=1):
            source_path = Path(raw_path)
            fraction = idx / max(total, 1)
            self.after(0, progress.update_progress, fraction, f"Processing {idx}/{total}", source_path.name)

            try:
                image_bgr = load_image_bgr(source_path)
            except ImageLoadError:
                rejected += 1
                continue

            face, reason = face_engine.detect_single_face(image_bgr)
            if face is None:
                rejected += 1
                continue

            accepted_embeddings.append(l2_normalize(face.embedding))
            accepted += 1
            copy_image_into_gallery(source_path, config.images_root, person.identity_id)

        if accepted > 0:
            database.add_embeddings(
                person_name=person.name,
                embeddings=accepted_embeddings,
                image_paths_added=accepted,
                representative_image=None,
                overwrite=False,
            )

        self.after(0, self._finish_update, progress, person.name, accepted, rejected)

    def _finish_update(self, progress: ProgressDialog, name: str, accepted: int, rejected: int) -> None:
        progress.close()
        self.refresh()
        if self._on_change:
            self._on_change()
        message = f"Added {accepted} new embedding(s) for '{name}'."
        if rejected:
            message += f" {rejected} image(s) were rejected (no face / multiple faces)."
        show_info_dialog(self, "Update Complete", message, success=accepted > 0)

    def _delete_person(self, person: PersonMetadata) -> None:
        def do_delete():
            self._context.database.delete_person(person.name)
            delete_person_directory(self._context.config.images_root, person.identity_id)
            self.refresh()
            if self._on_change:
                self._on_change()

        show_confirm_dialog(
            self, "Delete Person",
            f"Are you sure you want to permanently delete '{person.name}' and all associated images/embeddings?",
            on_confirm=do_delete,
            confirm_text="Delete",
        )
