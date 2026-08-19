"""
config.py
=========
Centralized, typed configuration management for VisionGuard AI.

The application never reads config.yaml directly outside of this module.
All other modules import `get_config()` to obtain a singleton, strongly
typed `AppConfig` instance. This keeps configuration access consistent,
avoids scattered YAML parsing, and makes it trivial to persist changes
made from the Settings page back to disk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml

# Project root is two levels up from this file: app/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AppSection:
    name: str = "VisionGuard AI"
    version: str = "1.0.0"
    window_width: int = 1200
    window_height: int = 750
    theme: str = "dark"
    color_theme: str = "blue"


@dataclass
class PathsSection:
    database_dir: str = "data/database"
    embeddings_file: str = "data/database/embeddings.pkl"
    metadata_file: str = "data/database/metadata.json"
    images_dir: str = "data/images"
    models_dir: str = "data/models"
    logs_dir: str = "data/logs"

    def resolve(self, relative: str) -> Path:
        """Resolve a relative path string against the project root."""
        return (PROJECT_ROOT / relative).resolve()


@dataclass
class FaceAnalysisSection:
    model_name: str = "buffalo_l"
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    detection_size: List[int] = field(default_factory=lambda: [640, 640])
    min_face_confidence: float = 0.5


@dataclass
class EnrollmentSection:
    min_images: int = 2
    recommended_images: int = 8
    max_images: int = 20


@dataclass
class RecognitionSection:
    similarity_threshold: float = 0.45
    recognition_smoothing_window: int = 5
    unknown_label: str = "Unknown"


@dataclass
class DistanceSection:
    known_face_width_cm: float = 14.0
    focal_length_px: float = 615.0
    too_close_max_cm: int = 50
    ideal_min_cm: int = 50
    ideal_max_cm: int = 100
    too_far_max_cm: int = 150


@dataclass
class CameraSection:
    device_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    target_fps: int = 30


@dataclass
class RobotSection:
    simulate_hardware: bool = True
    serial_port: str = "COM3"
    baud_rate: int = 9600
    center_dead_zone_px: int = 60
    forward_distance_cm: int = 100
    backward_distance_cm: int = 50
    command_timeout_ms: int = 750
    require_ack: bool = False
    ack_timeout_ms: int = 250


@dataclass
class LoggingSection:
    level: str = "INFO"
    max_bytes: int = 1_048_576
    backup_count: int = 5
    log_filename: str = "visionguard.log"


@dataclass
class AppConfig:
    app: AppSection = field(default_factory=AppSection)
    paths: PathsSection = field(default_factory=PathsSection)
    face_analysis: FaceAnalysisSection = field(default_factory=FaceAnalysisSection)
    enrollment: EnrollmentSection = field(default_factory=EnrollmentSection)
    recognition: RecognitionSection = field(default_factory=RecognitionSection)
    distance: DistanceSection = field(default_factory=DistanceSection)
    camera: CameraSection = field(default_factory=CameraSection)
    robot: RobotSection = field(default_factory=RobotSection)
    logging: LoggingSection = field(default_factory=LoggingSection)

    _source_path: Optional[Path] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from a YAML file, falling back to defaults
        for any missing keys so the application never crashes due to a
        partially-written config.yaml."""
        cfg_path = path or DEFAULT_CONFIG_PATH
        raw: dict = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

        instance = cls(
            app=AppSection(**raw.get("app", {})),
            paths=PathsSection(**raw.get("paths", {})),
            face_analysis=FaceAnalysisSection(**raw.get("face_analysis", {})),
            enrollment=EnrollmentSection(**raw.get("enrollment", {})),
            recognition=RecognitionSection(**raw.get("recognition", {})),
            distance=DistanceSection(**raw.get("distance", {})),
            camera=CameraSection(**raw.get("camera", {})),
            robot=RobotSection(**raw.get("robot", {})),
            logging=LoggingSection(**raw.get("logging", {})),
        )
        instance._source_path = cfg_path
        instance.validate()
        instance.ensure_directories()
        return instance

    def validate(self) -> None:
        """Reject unsafe or contradictory configuration before hardware starts."""
        if not 0.0 <= self.recognition.similarity_threshold <= 1.0:
            raise ValueError("recognition.similarity_threshold must be between 0 and 1")
        if not (0 < self.enrollment.min_images <= self.enrollment.recommended_images <= self.enrollment.max_images):
            raise ValueError("enrollment image limits must be positive and ordered")
        if not (0 <= self.distance.too_close_max_cm <= self.distance.ideal_min_cm <= self.distance.ideal_max_cm < self.distance.too_far_max_cm):
            raise ValueError("distance thresholds are contradictory")
        if not (self.robot.backward_distance_cm <= self.robot.forward_distance_cm):
            raise ValueError("robot backward distance must not exceed forward distance")
        if self.robot.command_timeout_ms <= 0 or self.robot.ack_timeout_ms <= 0:
            raise ValueError("robot timeouts must be positive")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the current configuration back to disk, preserving the
        original file location. Used by the Settings page."""
        self.validate()
        target = self._source_path or DEFAULT_CONFIG_PATH
        payload = {
            "app": asdict(self.app),
            "paths": asdict(self.paths),
            "face_analysis": asdict(self.face_analysis),
            "enrollment": asdict(self.enrollment),
            "recognition": asdict(self.recognition),
            "distance": asdict(self.distance),
            "camera": asdict(self.camera),
            "robot": asdict(self.robot),
            "logging": asdict(self.logging),
        }
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create all directories referenced by the config if they do not
        already exist. Called once at startup so the rest of the app can
        assume the folder structure is present."""
        for rel in (
            self.paths.database_dir,
            self.paths.images_dir,
            self.paths.models_dir,
            self.paths.logs_dir,
        ):
            self.paths.resolve(rel).mkdir(parents=True, exist_ok=True)

    @property
    def embeddings_path(self) -> Path:
        return self.paths.resolve(self.paths.embeddings_file)

    @property
    def metadata_path(self) -> Path:
        return self.paths.resolve(self.paths.metadata_file)

    @property
    def images_root(self) -> Path:
        return self.paths.resolve(self.paths.images_dir)

    @property
    def logs_root(self) -> Path:
        return self.paths.resolve(self.paths.logs_dir)

    @property
    def models_root(self) -> Path:
        return self.paths.resolve(self.paths.models_dir)


# -------------------------------------------------------------------------
# Singleton accessor
# -------------------------------------------------------------------------
_config_lock = threading.Lock()
_config_instance: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    """Return the process-wide singleton AppConfig instance, loading it
    from disk on first access (or when `reload=True` is requested after
    a settings change)."""
    global _config_instance
    with _config_lock:
        if _config_instance is None or reload:
            _config_instance = AppConfig.load()
        return _config_instance
