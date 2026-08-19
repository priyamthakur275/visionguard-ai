"""
test_web.py
===========
Automated integration tests for VisionGuard AI FastAPI web backend endpoints:
health check, statistics, person listing, frame recognition, and enrollment.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.web.server import app, database, config


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_index(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "VisionGuard AI" in response.text


def test_api_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "VisionGuard AI"

    # Alias /health test
    res_alias = client.get("/health")
    assert res_alias.status_code == 200
    assert res_alias.json()["status"] == "healthy"


def test_api_stats(client: TestClient):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_persons" in data
    assert "similarity_threshold" in data
    assert "distance_bands" in data

    # Alias /api/status test
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert "total_persons" in res_status.json()


def test_api_persons_lifecycle(client: TestClient, tmp_path: Path):
    # Ensure fresh DB state for test
    name = "WebTestUser"
    emb = np.ones(512, dtype=np.float32)
    identity_id = "11111111-2222-3333-4444-555555555555"

    database.add_embeddings(name, [emb], image_paths_added=1, representative_image="test.jpg", overwrite=True, identity_id=identity_id)

    # 1. List persons
    list_res = client.get("/api/persons")
    assert list_res.status_code == 200
    names = [p["name"] for p in list_res.json()["persons"]]
    assert name in names

    # 2. Delete person
    del_res = client.delete(f"/api/persons/{name}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 3. Verify deletion
    list_res2 = client.get("/api/persons")
    names2 = [p["name"] for p in list_res2.json()["persons"]]
    assert name not in names2


def test_api_recognize_frame_validation(client: TestClient):
    # Missing/invalid base64
    res = client.post("/api/recognize", json={"image_base64": "invalid_base64", "target_name": None})
    assert res.status_code == 400


def test_api_recognize_frame_with_mock_engine(client: TestClient):
    # Create valid blank image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    mock_engine = MagicMock()
    mock_res = SimpleNamespace(
        name="Priyam",
        is_known=True,
        similarity=0.85,
        bbox=(20, 20, 80, 80),
    )
    mock_face = SimpleNamespace(bbox=(20, 20, 80, 80))
    mock_engine.recognize_frame.return_value = [(mock_face, mock_res)]

    with patch("app.web.server.get_recognition_engine", return_value=mock_engine):
        response = client.post("/api/recognize", json={"image_base64": b64_str, "target_name": "Priyam"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["faces"]) == 1
        assert data["faces"][0]["name"] == "Priyam"
        assert data["robot"]["target_found"] is True
        assert data["robot"]["command"] in ["FORWARD", "BACKWARD", "STOP", "LEFT", "RIGHT"]


def test_api_enroll_validation(client: TestClient):
    # Reject without consent
    res = client.post(
        "/api/enroll",
        data={"name": "Alice", "consent": False},
        files=[("images", ("test1.jpg", b"fakebytes", "image/jpeg"))],
    )
    assert res.status_code == 400
    assert "Biometric consent" in res.json()["detail"]

    # Reject image count violation
    res2 = client.post(
        "/api/enroll",
        data={"name": "Alice", "consent": True},
        files=[("images", ("test1.jpg", b"fakebytes", "image/jpeg"))],
    )
    assert res2.status_code == 400
    assert "Please upload between" in res2.json()["detail"]
