"""
database_utils.py
==================
Persistence layer for the face database. Two files are maintained in
lockstep:

    embeddings.pkl  -> {person_name: List[np.ndarray]}   (joblib-serialized)
    metadata.json   -> {person_name: PersonMetadata}     (human-readable)

Every enrolled image contributes its OWN embedding row; embeddings are
never averaged together, per the enrollment specification. This module
is the single source of truth for reading/writing that data and is used
by both the Enroll and Registered Persons pages.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from app.config import AppConfig
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PersonMetadata:
    name: str
    identity_id: str = ""
    enrollment_date: str = ""
    last_updated: str = ""
    image_count: int = 0
    embedding_count: int = 0
    representative_image: Optional[str] = None  # relative path for thumbnails


@dataclass
class EnrollmentResult:
    person_name: str
    images_processed: int
    images_accepted: int
    images_rejected: int
    rejection_reasons: List[str] = field(default_factory=list)


class FaceDatabase:
    """Thread-safe facade over the embeddings + metadata files."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._lock = threading.RLock()
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._metadata: Dict[str, PersonMetadata] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            embeddings_path = self._config.embeddings_path
            metadata_path = self._config.metadata_path

            if embeddings_path.exists():
                try:
                    self._embeddings = joblib.load(embeddings_path)
                    logger.info("Loaded embeddings database (%d identities).", len(self._embeddings))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to load embeddings database, starting fresh: %s", exc)
                    self._embeddings = {}
            else:
                self._embeddings = {}

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as handle:
                        raw = json.load(handle)
                    self._metadata = {}
                    valid_keys = {
                        "name", "identity_id", "enrollment_date", "last_updated",
                        "image_count", "embedding_count", "representative_image"
                    }
                    for name, payload in raw.items():
                        filtered = {k: v for k, v in payload.items() if k in valid_keys}
                        filtered.setdefault("name", name)
                        filtered.setdefault("identity_id", str(uuid.uuid4()))
                        filtered.setdefault("enrollment_date", datetime.now().isoformat(timespec="seconds"))
                        filtered.setdefault("last_updated", filtered["enrollment_date"])
                        filtered.setdefault("image_count", 0)
                        filtered.setdefault("embedding_count", len(self._embeddings.get(name, [])))
                        filtered.setdefault("representative_image", None)
                        self._metadata[name] = PersonMetadata(**filtered)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to load metadata database, starting fresh: %s", exc)
                    self._metadata = {}
            else:
                self._metadata = {}

    def _persist(self) -> None:
        """Atomically persist both embeddings and metadata to disk."""
        with self._lock:
            embeddings_path = self._config.embeddings_path
            metadata_path = self._config.metadata_path
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_embeddings = embeddings_path.with_suffix(".pkl.tmp")
            joblib.dump(self._embeddings, tmp_embeddings)
            tmp_embeddings.replace(embeddings_path)

            tmp_metadata = metadata_path.with_suffix(".json.tmp")
            serializable = {name: asdict(meta) for name, meta in self._metadata.items()}
            with open(tmp_metadata, "w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2)
            tmp_metadata.replace(metadata_path)

            logger.debug("Persisted face database to disk.")

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def person_exists(self, person_name: str) -> bool:
        with self._lock:
            return person_name in self._embeddings

    def list_persons(self) -> List[PersonMetadata]:
        with self._lock:
            return list(self._metadata.values())

    def get_metadata(self, person_name: str) -> Optional[PersonMetadata]:
        with self._lock:
            return self._metadata.get(person_name)

    def get_embeddings_for(self, person_name: str) -> List[np.ndarray]:
        with self._lock:
            return list(self._embeddings.get(person_name, []))

    def total_persons(self) -> int:
        with self._lock:
            return len(self._embeddings)

    def stacked_gallery(self):
        """Return (embeddings_matrix, labels_list) across ALL persons,
        suitable for embedding_utils.best_match()."""
        with self._lock:
            all_embeddings: List[np.ndarray] = []
            all_labels: List[str] = []
            for name, vectors in self._embeddings.items():
                for vector in vectors:
                    all_embeddings.append(vector)
                    all_labels.append(name)
            if not all_embeddings:
                return np.empty((0, 512), dtype=np.float32), []
            return np.vstack(all_embeddings).astype(np.float32), all_labels

    def last_enrollment_timestamp(self) -> Optional[str]:
        with self._lock:
            if not self._metadata:
                return None
            return max(meta.last_updated for meta in self._metadata.values())

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------
    def add_embeddings(
        self,
        person_name: str,
        embeddings: List[np.ndarray],
        image_paths_added: int,
        representative_image: Optional[str],
        overwrite: bool,
        identity_id: Optional[str] = None,
    ) -> None:
        """Add embeddings for a person. If overwrite=True, existing
        embeddings/metadata for that person are replaced entirely;
        otherwise new embeddings are appended to the existing gallery."""
        with self._lock:
            now = datetime.now().isoformat(timespec="seconds")

            if overwrite or person_name not in self._embeddings:
                self._embeddings[person_name] = list(embeddings)
                new_image_count = image_paths_added
                enrollment_date = now
            else:
                self._embeddings[person_name].extend(embeddings)
                existing = self._metadata.get(person_name)
                new_image_count = (existing.image_count if existing else 0) + image_paths_added
                enrollment_date = existing.enrollment_date if existing else now

            resolved_id = (
                identity_id
                or (self._metadata[person_name].identity_id if person_name in self._metadata else str(uuid.uuid4()))
            )

            self._metadata[person_name] = PersonMetadata(
                name=person_name,
                identity_id=resolved_id,
                enrollment_date=enrollment_date,
                last_updated=now,
                image_count=new_image_count,
                embedding_count=len(self._embeddings[person_name]),
                representative_image=representative_image
                or (self._metadata[person_name].representative_image if person_name in self._metadata else None),
            )
            self._persist()
            logger.info(
                "Saved %d embedding(s) for '%s' (overwrite=%s). Total embeddings now: %d",
                len(embeddings), person_name, overwrite, len(self._embeddings[person_name]),
            )

    def delete_person(self, person_name: str) -> None:
        with self._lock:
            self._embeddings.pop(person_name, None)
            self._metadata.pop(person_name, None)
            self._persist()
            logger.info("Deleted person '%s' from database.", person_name)

    def identity_id_for(self, person_name: str) -> Optional[str]:
        """Return the opaque UUID used for the person's image directory."""
        with self._lock:
            metadata = self._metadata.get(person_name)
            return metadata.identity_id if metadata else None


_db_instance: Optional[FaceDatabase] = None
_db_lock = threading.Lock()


def get_database(config: AppConfig) -> FaceDatabase:
    """Process-wide singleton accessor for the FaceDatabase."""
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            _db_instance = FaceDatabase(config)
        return _db_instance
