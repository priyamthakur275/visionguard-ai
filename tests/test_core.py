from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config import AppConfig, DistanceSection, EnrollmentSection, RecognitionSection, RobotSection
from app.core.database_utils import FaceDatabase, PersonMetadata
from app.core.distance_utils import (
    DistanceZone,
    bbox_pixel_width,
    calibrate_focal_length,
    classify_distance,
    estimate_distance_cm,
)
from app.core.embedding_utils import (
    batch_cosine_similarity,
    best_match,
    cosine_similarity,
    l2_normalize,
)
from app.core.file_utils import (
    copy_image_into_gallery,
    count_all_images,
    delete_person_directory,
    ensure_dir,
    generate_unique_filename,
    is_supported_image,
    list_person_images,
    person_image_dir,
    sanitize_person_name,
)
from app.core.image_utils import (
    ImageLoadError,
    bgr_to_rgb,
    crop_box,
    load_image_bgr,
    make_placeholder_thumbnail,
    resize_keep_aspect,
    thumbnail_from_path,
)
from app.core.recognition_utils import LabelSmoother
from app.core.recognition_worker import LatestFrameRecognitionWorker
from app.core.robot_utils import (
    RobotCommand,
    RobotController,
    decide_command,
    select_recognized_target,
)
from app.core.tracking_utils import CentroidTracker


# =====================================================================
# Configuration Tests
# =====================================================================
def test_default_config_validates():
    config = AppConfig()
    config.validate()
    assert config.app.name == "VisionGuard AI"
    assert config.robot.simulate_hardware is True


def test_config_similarity_threshold_validation():
    config = AppConfig()
    config.recognition.similarity_threshold = -0.1
    with pytest.raises(ValueError, match="similarity_threshold"):
        config.validate()

    config.recognition.similarity_threshold = 1.1
    with pytest.raises(ValueError, match="similarity_threshold"):
        config.validate()


def test_config_enrollment_limits_validation():
    config = AppConfig()
    config.enrollment.min_images = 10
    config.enrollment.recommended_images = 5
    with pytest.raises(ValueError, match="enrollment image limits"):
        config.validate()


def test_config_distance_thresholds_validation():
    config = AppConfig()
    config.distance.ideal_max_cm = 40
    config.distance.too_close_max_cm = 50
    with pytest.raises(ValueError, match="distance thresholds"):
        config.validate()


def test_config_robot_bounds_validation():
    config = AppConfig()
    config.robot.forward_distance_cm = 30
    config.robot.backward_distance_cm = 60
    with pytest.raises(ValueError, match="robot backward distance"):
        config.validate()

    config2 = AppConfig()
    config2.robot.command_timeout_ms = 0
    with pytest.raises(ValueError, match="robot timeouts"):
        config2.validate()


def test_config_save_and_load_roundtrip(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    config = AppConfig()
    config._source_path = cfg_file
    config.app.name = "VisionGuard Custom"
    config.distance.ideal_max_cm = 120
    config.distance.too_far_max_cm = 180
    config.robot.forward_distance_cm = 120
    config.save()

    loaded = AppConfig.load(cfg_file)
    assert loaded.app.name == "VisionGuard Custom"
    assert loaded.distance.ideal_max_cm == 120
    assert loaded.distance.too_far_max_cm == 180


# =====================================================================
# Distance Estimation Tests
# =====================================================================
def test_distance_policy_classification():
    config = AppConfig()
    assert classify_distance(-5, config).zone is DistanceZone.OUT_OF_RANGE
    assert classify_distance(50, config).zone is DistanceZone.TOO_CLOSE
    assert classify_distance(51, config).zone is DistanceZone.IDEAL
    assert classify_distance(100, config).zone is DistanceZone.IDEAL
    assert classify_distance(101, config).zone is DistanceZone.TOO_FAR
    assert classify_distance(150, config).zone is DistanceZone.TOO_FAR
    assert classify_distance(151, config).zone is DistanceZone.OUT_OF_RANGE


def test_estimate_distance_cm_and_bbox_width():
    config = AppConfig()
    bbox = (100, 100, 200, 250)
    width = bbox_pixel_width(bbox)
    assert width == 100.0

    dist = estimate_distance_cm(width, config)
    expected = (config.distance.known_face_width_cm * config.distance.focal_length_px) / 100.0
    assert pytest.approx(dist) == expected

    assert estimate_distance_cm(0, config) == -1.0
    assert estimate_distance_cm(-10, config) == -1.0


def test_calibrate_focal_length():
    focal = calibrate_focal_length(face_pixel_width=100.0, known_distance_cm=86.1, known_face_width_cm=14.0)
    assert pytest.approx(focal, rel=1e-2) == 615.0
    with pytest.raises(ValueError):
        calibrate_focal_length(0, 50, 14)


# =====================================================================
# Embedding Math Tests
# =====================================================================
def test_l2_normalize():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(vec)
    assert pytest.approx(np.linalg.norm(normed)) == 1.0
    assert pytest.approx(normed[0]) == 0.6
    assert pytest.approx(normed[1]) == 0.8

    zero_vec = np.zeros(512, dtype=np.float32)
    normed_zero = l2_normalize(zero_vec)
    assert np.allclose(normed_zero, 0.0)


def test_cosine_similarity():
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([1.0, 0.0], dtype=np.float32)
    vec_c = np.array([0.0, 1.0], dtype=np.float32)
    vec_d = np.array([-1.0, 0.0], dtype=np.float32)

    assert pytest.approx(cosine_similarity(vec_a, vec_b)) == 1.0
    assert pytest.approx(cosine_similarity(vec_a, vec_c)) == 0.0
    assert pytest.approx(cosine_similarity(vec_a, vec_d)) == -1.0


def test_batch_cosine_similarity():
    query = np.array([1.0, 0.0], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    scores = batch_cosine_similarity(query, gallery)
    assert len(scores) == 3
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 0.0
    assert pytest.approx(scores[2], abs=1e-3) == 0.7071

    empty_scores = batch_cosine_similarity(query, np.empty((0, 2), dtype=np.float32))
    assert len(empty_scores) == 0


def test_best_match():
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    label, score = best_match(np.asarray([0.9, 0.1], dtype=np.float32), gallery, ["Alice", "Bob"])
    assert label == "Alice"
    assert score > 0.9

    empty_label, empty_score = best_match(np.asarray([1.0, 0.0]), np.empty((0, 2)), [])
    assert empty_label == ""
    assert empty_score == -1.0


# =====================================================================
# Filesystem & Storage Tests
# =====================================================================
def test_identity_directory_uuid_enforcement(tmp_path: Path):
    valid_uuid = "c7b7da73-5b64-4e18-97df-f3a7751b4ad2"
    path = person_image_dir(tmp_path, valid_uuid)
    assert path == tmp_path / valid_uuid
    assert path.exists()

    with pytest.raises(ValueError, match="identity_id must be a UUID"):
        person_image_dir(tmp_path, "John Doe")


def test_file_utils_helpers(tmp_path: Path):
    assert is_supported_image(Path("photo.JPG")) is True
    assert is_supported_image(Path("photo.png")) is True
    assert is_supported_image(Path("data.txt")) is False

    name = generate_unique_filename("sample.png")
    assert name.endswith(".png")

    assert sanitize_person_name("Dr. Jane Doe / 01") == "Dr__Jane_Doe___01"


def test_gallery_image_management(tmp_path: Path):
    images_root = tmp_path / "images"
    identity_id = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"

    src_img = tmp_path / "test.jpg"
    src_img.write_bytes(b"fake image data")

    dest = copy_image_into_gallery(src_img, images_root, identity_id)
    assert dest.exists()
    assert dest.parent == images_root / identity_id

    listed = list_person_images(images_root, identity_id)
    assert len(listed) == 1
    assert listed[0] == dest
    assert count_all_images(images_root) == 1

    delete_person_directory(images_root, identity_id)
    assert not (images_root / identity_id).exists()
    assert count_all_images(images_root) == 0


# =====================================================================
# Image Processing Helpers Tests
# =====================================================================
def test_crop_box_and_resize():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop_box(img, (20, 20, 60, 60), margin=0.0)
    assert cropped.shape == (40, 40, 3)

    # Edge clipping test
    cropped_clipped = crop_box(img, (0, 0, 50, 50), margin=0.5)
    assert cropped_clipped.shape[0] <= 100
    assert cropped_clipped.shape[1] <= 100

    resized = resize_keep_aspect(img, 50)
    assert resized.shape == (50, 50, 3)


def test_bgr_to_rgb_and_thumbnails():
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    bgr[0, 0] = [255, 0, 0]  # Blue pixel
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0, 0] == 0 and rgb[0, 0, 2] == 255  # Red pixel in RGB

    thumb = make_placeholder_thumbnail(size=(48, 48))
    assert thumb.size == (48, 48)

    fallback = thumbnail_from_path(None, size=(32, 32))
    assert fallback.size == (32, 32)


def test_load_image_bgr_nonexistent_raises(tmp_path: Path):
    with pytest.raises(ImageLoadError):
        load_image_bgr(tmp_path / "missing.jpg")


# =====================================================================
# Tracking & Label Smoothing Tests
# =====================================================================
def test_label_smoother():
    smoother = LabelSmoother(window=3)
    assert smoother.update("Priyam") == "Priyam"
    assert smoother.update("Unknown") == "Priyam"  # 2:1 majority
    assert smoother.update("Unknown") == "Unknown"  # 2:1 majority for Unknown


def test_centroid_tracker_lifecycle():
    tracker = CentroidTracker(max_distance_px=50.0, max_missed_frames=2, smoothing_window=3)
    
    # Frame 1: One face
    tracks1 = tracker.update([(100, 100, 150, 150)])
    assert len(tracks1) == 1
    track_id = list(tracks1.keys())[0]

    # Frame 2: Face moved slightly
    tracks2 = tracker.update([(105, 102, 155, 152)])
    assert track_id in tracks2

    # Frame 3 & 4: Face disappeared
    tracker.update([])
    tracker.update([])
    # Frame 5: Should be purged
    tracks5 = tracker.update([])
    assert len(tracks5) == 0


def test_largest_active_track():
    tracker = CentroidTracker()
    tracker.update([(10, 10, 30, 30), (50, 50, 150, 150)])  # Area 400 vs 10000
    largest = tracker.largest_active_track()
    assert largest is not None
    assert largest[1] == (50, 50, 150, 150)


# =====================================================================
# Robot Logic & Decision Tests
# =====================================================================
def test_decide_command_horizontal_alignment():
    config = AppConfig()
    # Dead zone is 60px. Frame center = 480 (for 960 width).
    # Target at x=300 -> Left by 180px
    decision_left = decide_command(960, 300, 75, config)
    assert decision_left.command is RobotCommand.LEFT

    # Target at x=600 -> Right by 120px
    decision_right = decide_command(960, 600, 75, config)
    assert decision_right.command is RobotCommand.RIGHT


def test_decide_command_distance_regulation():
    config = AppConfig()
    # Centered at x=480
    assert decide_command(960, 480, 110, config).command is RobotCommand.FORWARD
    assert decide_command(960, 480, 40, config).command is RobotCommand.BACKWARD
    assert decide_command(960, 480, 75, config).command is RobotCommand.STOP
    assert decide_command(960, 480, -1, config).command is RobotCommand.STOP


def test_select_recognized_target():
    face_small = SimpleNamespace(bbox=(0, 0, 10, 10))
    face_large = SimpleNamespace(bbox=(0, 0, 30, 30))
    face_other = SimpleNamespace(bbox=(0, 0, 50, 50))
    face_unknown = SimpleNamespace(bbox=(0, 0, 100, 100))

    make_res = lambda name, known, bbox: SimpleNamespace(name=name, is_known=known, bbox=bbox)
    candidates = [
        (face_small, make_res("Priyam", True, face_small.bbox)),
        (face_large, make_res("Priyam", True, face_large.bbox)),
        (face_other, make_res("Alice", True, face_other.bbox)),
        (face_unknown, make_res("Unknown", False, face_unknown.bbox)),
    ]

    # Must pick largest matching known face
    target = select_recognized_target(candidates, "Priyam")
    assert target is face_large

    # Unknown must never match
    assert select_recognized_target(candidates, "Unknown") is None
    assert select_recognized_target(candidates, "NonExistent") is None


def test_robot_controller_simulation_mode():
    config = AppConfig()
    config.robot.simulate_hardware = True
    controller = RobotController(config)
    assert controller.connect() is True
    assert controller.is_connected is True

    controller.send_command(RobotCommand.FORWARD)
    controller.send_command(RobotCommand.FORWARD)  # Deduplicated
    controller.send_command(RobotCommand.STOP)
    controller.disconnect()
    assert controller.is_connected is False


# =====================================================================
# Database Tests
# =====================================================================
def test_face_database_crud(tmp_path: Path):
    config = AppConfig()
    config.paths.database_dir = str(tmp_path / "database")
    config.paths.embeddings_file = str(tmp_path / "database" / "embeddings.pkl")
    config.paths.metadata_file = str(tmp_path / "database" / "metadata.json")
    config.paths.images_dir = str(tmp_path / "images")

    db = FaceDatabase(config)
    assert db.total_persons() == 0

    emb1 = np.ones(512, dtype=np.float32)
    emb2 = np.ones(512, dtype=np.float32) * 2
    identity_id = "550e8400-e29b-41d4-a716-446655440000"

    db.add_embeddings("Priyam", [emb1, emb2], image_paths_added=2, representative_image="priyam.jpg", overwrite=True, identity_id=identity_id)
    assert db.person_exists("Priyam") is True
    assert db.total_persons() == 1
    assert db.identity_id_for("Priyam") == identity_id

    meta = db.get_metadata("Priyam")
    assert meta is not None
    assert meta.embedding_count == 2
    assert meta.image_count == 2

    # Stacked gallery
    matrix, labels = db.stacked_gallery()
    assert matrix.shape == (2, 512)
    assert labels == ["Priyam", "Priyam"]

    # Append test
    emb3 = np.ones(512, dtype=np.float32) * 3
    db.add_embeddings("Priyam", [emb3], image_paths_added=1, representative_image=None, overwrite=False)
    assert len(db.get_embeddings_for("Priyam")) == 3
    assert db.get_metadata("Priyam").image_count == 3

    # Delete test
    db.delete_person("Priyam")
    assert db.person_exists("Priyam") is False
    assert db.total_persons() == 0


def test_face_database_handles_corrupt_or_legacy_metadata(tmp_path: Path):
    config = AppConfig()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    config.paths.database_dir = str(db_dir)
    config.paths.embeddings_file = str(db_dir / "embeddings.pkl")
    config.paths.metadata_file = str(db_dir / "metadata.json")

    # Write legacy json without identity_id or with extra fields
    legacy_data = {
        "TestUser": {
            "name": "TestUser",
            "enrollment_date": "2026-01-01T00:00:00",
            "image_count": 5,
            "extra_field": "legacy"
        }
    }
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)

    db = FaceDatabase(config)
    meta = db.get_metadata("TestUser")
    assert meta is not None
    assert meta.name == "TestUser"
    assert len(meta.identity_id) > 0  # auto-generated UUID


# =====================================================================
# Recognition Worker Tests
# =====================================================================
def test_recognition_worker_execution():
    mock_engine = MagicMock()
    mock_engine.recognize_frame.return_value = [("face_obj", SimpleNamespace(name="Priyam", is_known=True, bbox=(0, 0, 10, 10)))]

    worker = LatestFrameRecognitionWorker(mock_engine)
    worker.start()

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    worker.submit(dummy_frame)

    # Wait briefly for worker thread to process
    result = None
    for _ in range(20):
        time.sleep(0.05)
        result = worker.latest_result()
        if result is not None:
            break

    worker.stop()
    assert result is not None
    assert result.error is None
    assert len(result.results) == 1
    assert result.inference_fps > 0

# =====================================================================
# Additional Robustness & Edge-Case Tests
# =====================================================================
def test_config_missing_file_falls_back_to_defaults(tmp_path: Path):
    missing_file = tmp_path / "non_existent_config.yaml"
    cfg = AppConfig.load(missing_file)
    assert cfg.app.name == "VisionGuard AI"
    assert cfg.distance.known_face_width_cm == 14.0


def test_distance_zone_display_and_negative_width():
    config = AppConfig()
    reading = classify_distance(75.0, config)
    assert reading.zone is DistanceZone.IDEAL
    assert reading.label == "IDEAL DISTANCE"
    assert reading.color_hex == "#2ECC71"
    assert reading.distance_cm == 75.0

    assert estimate_distance_cm(0.0, config) == -1.0
    assert estimate_distance_cm(-50.0, config) == -1.0


def test_centroid_tracker_multiple_tracks_gating():
    tracker = CentroidTracker(max_distance_px=30.0, max_missed_frames=1, smoothing_window=2)
    # 2 faces far apart
    tracks = tracker.update([(0, 0, 10, 10), (200, 200, 210, 210)])
    assert len(tracks) == 2

    # Move each slightly
    tracks2 = tracker.update([(2, 2, 12, 12), (202, 202, 212, 212)])
    assert len(tracks2) == 2
    assert set(tracks.keys()) == set(tracks2.keys())

    # Find track ID for bbox
    t_id = tracker.find_track_id_for_bbox((2, 2, 12, 12))
    assert t_id in tracks2


def test_robot_controller_ack_and_timeout():
    config = AppConfig()
    config.robot.simulate_hardware = False
    controller = RobotController(config)
    # In absence of real serial port / mock, connect should safely return False
    config.robot.serial_port = "NON_EXISTENT_COM999"
    connected = controller.connect()
    assert connected is False
    assert controller.is_connected is False


def test_label_smoother_edge_cases():
    smoother = LabelSmoother(window=1)
    assert smoother.update("Alice") == "Alice"
    assert smoother.update("Bob") == "Bob"

    smoother_large = LabelSmoother(window=10)
    for _ in range(6):
        smoother_large.update("Known")
    for _ in range(4):
        smoother_large.update("Unknown")
    assert smoother_large.update("Known") == "Known"
