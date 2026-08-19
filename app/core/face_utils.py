"""
face_utils.py
=============
Thin, defensive wrapper around InsightFace's Buffalo_L model pack.

IMPORTANT: Buffalo_L is a pretrained model. It is NEVER retrained or
fine-tuned by this application. It is used exclusively, out-of-the-box,
for:
    1. Face detection (RetinaFace-based detector bundled in buffalo_l)
    2. Face alignment (5-point landmark warp, handled internally by
       InsightFace's FaceAnalysis pipeline)
    3. ArcFace embedding extraction (512-D feature vector)

"Training" in this project only ever refers to enrolling a person, i.e.
generating and persisting embeddings - never touching model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import threading

from app.config import AppConfig
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedFace:
    """A single detected face with everything downstream modules need."""
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2) in pixel coords
    confidence: float
    landmarks: np.ndarray                  # 5x2 aligned landmark points
    embedding: np.ndarray                  # 512-D raw ArcFace embedding


class FaceAnalysisEngine:
    """Lazily-initialized singleton wrapper around insightface.app.FaceAnalysis.

    Lazy initialization matters because model loading (~1-2s and disk
    I/O for ONNX weights) should not block the UI from opening; it is
    triggered on first real use (enrollment or live recognition start).
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._analyzer = None  # type: Optional["insightface.app.FaceAnalysis"]
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._analyzer is not None:
                return
            try:
                # Imported lazily so the rest of the app can start even if
                # insightface/onnxruntime are still being installed/downloaded.
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError(
                    "InsightFace is not installed. Run: pip install insightface onnxruntime"
                ) from exc

            logger.info("Loading InsightFace model pack '%s' ...", self._config.face_analysis.model_name)
            try:
                analyzer = FaceAnalysis(
                    name=self._config.face_analysis.model_name,
                    root=str(self._config.models_root),
                    providers=self._config.face_analysis.providers,
                )
                det_w, det_h = self._config.face_analysis.detection_size
                analyzer.prepare(ctx_id=0, det_size=(det_w, det_h))
            except Exception as exc:  # model download / ONNX provider failures
                logger.exception("InsightFace initialization failed")
                raise RuntimeError(
                    "Could not initialize the InsightFace model. Check internet access for the initial "
                    "model download, ONNX Runtime compatibility, and the configured execution provider."
                ) from exc
            self._analyzer = analyzer
            logger.info("InsightFace model pack loaded successfully.")

    def ensure_loaded(self) -> None:
        """Eagerly validate the model before a camera session begins."""
        self._ensure_loaded()

    def detect_faces(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """Run detection + alignment + embedding extraction on a BGR
        image, returning one DetectedFace per face found. Buffalo_L
        performs alignment and embedding internally as part of `.get()`.
        """
        self._ensure_loaded()
        if image_bgr is None or image_bgr.size == 0:
            return []

        raw_faces = self._analyzer.get(image_bgr)
        min_conf = self._config.face_analysis.min_face_confidence

        results: List[DetectedFace] = []
        for face in raw_faces:
            det_score = float(getattr(face, "det_score", 1.0))
            if det_score < min_conf:
                continue
            bbox = tuple(int(v) for v in face.bbox)  # (x1, y1, x2, y2)
            landmarks = np.array(face.kps, dtype=np.float32) if hasattr(face, "kps") else np.empty((0, 2))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            results.append(
                DetectedFace(
                    bbox=bbox,  # type: ignore[arg-type]
                    confidence=det_score,
                    landmarks=landmarks,
                    embedding=embedding,
                )
            )
        return results

    def detect_single_face(self, image_bgr: np.ndarray) -> Tuple[Optional[DetectedFace], str]:
        """Enforce the enrollment rule: exactly one face per image.

        Returns (face, "") on success, or (None, reason) on rejection,
        where reason is one of: 'no_face', 'multiple_faces'.
        """
        faces = self.detect_faces(image_bgr)
        if len(faces) == 0:
            return None, "no_face"
        if len(faces) > 1:
            return None, "multiple_faces"
        return faces[0], ""


_engine_instance: Optional[FaceAnalysisEngine] = None


def get_face_engine(config: AppConfig) -> FaceAnalysisEngine:
    """Process-wide singleton accessor for the FaceAnalysisEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FaceAnalysisEngine(config)
    return _engine_instance
