"""
file_utils.py
=============
Small, dependency-free filesystem helpers used across the application:
safe directory creation, unique filename generation, and person image
folder management. Kept separate from image_utils.py (which deals with
pixel data) to maintain single-responsibility boundaries.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List

from app.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_supported_image(path: Path) -> bool:
    """Return True if the file extension is a supported image type."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def generate_unique_filename(original_name: str) -> str:
    """Generate a collision-free filename that preserves the original
    extension, e.g. 'a1b2c3d4.jpg'."""
    suffix = Path(original_name).suffix.lower() or ".jpg"
    return f"{uuid.uuid4().hex}{suffix}"


def person_image_dir(images_root: Path, identity_id: str) -> Path:
    """Return the per-identity directory. Display names never form paths."""
    try:
        safe_identity = str(uuid.UUID(identity_id))
    except (ValueError, TypeError) as exc:
        raise ValueError("identity_id must be a UUID") from exc
    return ensure_dir(images_root / safe_identity)


def sanitize_person_name(name: str) -> str:
    """Convert a display name into a filesystem-safe folder name."""
    cleaned = "".join(c if (c.isalnum() or c in (" ", "_", "-")) else "_" for c in name)
    return cleaned.strip().replace(" ", "_")


def copy_image_into_gallery(source_path: Path, images_root: Path, identity_id: str) -> Path:
    """Copy an externally-selected image into the person's managed
    gallery folder under a unique filename, returning the new path."""
    destination_dir = person_image_dir(images_root, identity_id)
    destination_path = destination_dir / generate_unique_filename(source_path.name)
    shutil.copy2(source_path, destination_path)
    logger.info("Copied image '%s' -> '%s'", source_path, destination_path)
    return destination_path


def delete_person_directory(images_root: Path, identity_id: str) -> None:
    """Remove a person's entire image folder (used on delete)."""
    try:
        folder = images_root / str(uuid.UUID(identity_id))
    except (ValueError, TypeError):
        logger.error("Refusing to delete non-UUID identity directory")
        return
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
        logger.info("Deleted image directory for identity '%s'", str(uuid.UUID(identity_id)))


def list_person_images(images_root: Path, identity_id: str) -> List[Path]:
    """Return all supported image paths stored for a given person."""
    try:
        folder = images_root / str(uuid.UUID(identity_id))
    except (ValueError, TypeError):
        return []
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if is_supported_image(p))


def count_all_images(images_root: Path) -> int:
    """Count every supported image file stored across all persons."""
    if not images_root.exists():
        return 0
    return sum(1 for p in images_root.rglob("*") if p.is_file() and is_supported_image(p))
