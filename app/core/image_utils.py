"""
image_utils.py
==============
Low-level image I/O helpers built on OpenCV/NumPy/Pillow. Responsible
for safely loading images from disk (including handling of corrupted
files and unicode paths on Windows), color-space conversions, and
converting frames to formats CustomTkinter can render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from app.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None


class ImageLoadError(Exception):
    """Raised when an image file cannot be decoded."""


def load_image_bgr(path: Path) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array (OpenCV convention).

    Uses np.fromfile + cv2.imdecode instead of cv2.imread so that
    unicode/non-ASCII file paths (common on Windows) decode correctly.
    Raises ImageLoadError for missing or corrupted files instead of
    letting OpenCV silently return None.
    """
    if not path.exists():
        raise ImageLoadError(f"Image file does not exist: {path}")
    if cv2 is None:
        raise ImageLoadError("OpenCV is not installed; cannot decode images.")
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - we want to wrap any decode failure
        raise ImageLoadError(f"Failed to read image '{path}': {exc}") from exc

    if image is None:
        raise ImageLoadError(f"Image is corrupted or unsupported: {path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB."""
    if cv2 is None:
        return image_bgr[..., ::-1].copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize an image to a target width while preserving aspect ratio."""
    height, width = image.shape[:2]
    if width == 0:
        return image
    scale = target_width / float(width)
    new_size = (target_width, max(1, int(height * scale)))
    if cv2 is None:
        pil = Image.fromarray(image)
        resized = pil.resize(new_size, Image.BILINEAR)
        return np.array(resized)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame into a Pillow Image (RGB) for display
    inside CustomTkinter widgets (CTkImage requires PIL.Image)."""
    rgb = bgr_to_rgb(image_bgr)
    return Image.fromarray(rgb)


def crop_box(image: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.0) -> np.ndarray:
    """Crop a bounding box (x1, y1, x2, y2) from an image, optionally
    expanding it by a margin fraction, clipped to image bounds."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * margin))
    y1 = max(0, int(y1 - box_h * margin))
    x2 = min(width, int(x2 + box_w * margin))
    y2 = min(height, int(y2 + box_h * margin))
    return image[y1:y2, x1:x2]


def make_placeholder_thumbnail(size: Tuple[int, int] = (96, 96)) -> Image.Image:
    """Generate a neutral placeholder thumbnail for persons without a
    representative photo available (used defensively in the UI)."""
    return Image.new("RGB", size, color=(58, 62, 70))


def thumbnail_from_path(path: Optional[Path], size: Tuple[int, int] = (96, 96)) -> Image.Image:
    """Load an image and produce a square-cropped thumbnail; falls back
    to a placeholder if the file is missing or unreadable."""
    if path is None:
        return make_placeholder_thumbnail(size)
    try:
        bgr = load_image_bgr(path)
    except ImageLoadError as exc:
        logger.warning("Thumbnail load failed: %s", exc)
        return make_placeholder_thumbnail(size)

    height, width = bgr.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    square = bgr[y0:y0 + side, x0:x0 + side]
    pil_image = bgr_to_pil(square)
    pil_image = pil_image.resize(size, Image.LANCZOS)
    return pil_image
