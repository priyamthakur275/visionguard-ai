"""
server.py
=========
FastAPI backend service for VisionGuard AI web demo.
Provides RESTful APIs and real-time WebSocket pipelines for face recognition,
identity enrollment, distance estimation, and robot tracking simulation.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from app.config import AppConfig
from app.core.database_utils import FaceDatabase
from app.core.distance_utils import bbox_pixel_width, classify_distance, estimate_distance_cm
from app.core.face_utils import FaceAnalysisEngine
from app.core.file_utils import copy_image_into_gallery, delete_person_directory, person_image_dir
from app.core.recognition_utils import RecognitionEngine
from app.core.robot_utils import RobotCommand, decide_command, select_recognized_target
from app.core.tracking_utils import CentroidTracker
from app.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="VisionGuard AI API",
    description="REST & WebSocket API for Face Recognition and Robot Tracking",
    version="1.0.0",
)

# CORS middleware for local testing and cloud cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration and initialize storage
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
config = AppConfig.load(CONFIG_PATH)

database = FaceDatabase(config)
face_engine = FaceAnalysisEngine(config)
recognition_engine: Optional[RecognitionEngine] = None
tracker = CentroidTracker(smoothing_window=config.recognition.recognition_smoothing_window)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
STATIC_DIR = Path(__file__).resolve().parent / "static"

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_recognition_engine() -> RecognitionEngine:
    global recognition_engine
    if recognition_engine is None:
        face_engine.ensure_loaded()
        recognition_engine = RecognitionEngine(config, face_engine, database)
    return recognition_engine


# ----------------------------------------------------------------------
# Web Interface Route
# ----------------------------------------------------------------------
@app.get("/")
async def serve_index():
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "VisionGuard AI API is running. index.html not found."})


# ----------------------------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": config.app.name,
        "version": config.app.version,
        "model_loaded": face_engine._analyzer is not None,
        "total_identities": database.total_persons(),
    }


@app.get("/api/status")
@app.get("/api/stats")
async def get_stats():
    matrix, labels = database.stacked_gallery()
    return {
        "total_persons": database.total_persons(),
        "total_embeddings": matrix.shape[0],
        "model_name": config.face_analysis.model_name,
        "similarity_threshold": config.recognition.similarity_threshold,
        "distance_bands": {
            "too_close_cm": config.distance.too_close_max_cm,
            "ideal_min_cm": config.distance.ideal_min_cm,
            "ideal_max_cm": config.distance.ideal_max_cm,
            "too_far_cm": config.distance.too_far_max_cm,
        },
        "robot_settings": {
            "forward_cm": config.robot.forward_distance_cm,
            "backward_cm": config.robot.backward_distance_cm,
            "dead_zone_px": config.robot.center_dead_zone_px,
        },
    }


@app.get("/api/persons")
async def list_persons():
    persons_meta = []
    for meta in database.list_persons():
        persons_meta.append({
            "name": meta.name,
            "identity_id": meta.identity_id,
            "enrollment_date": meta.enrollment_date,
            "image_count": meta.image_count,
            "embedding_count": meta.embedding_count,
            "representative_image": meta.representative_image,
        })
    return {"persons": persons_meta}


@app.post("/api/enroll")
async def enroll_person(
    name: str = Form(...),
    consent: bool = Form(...),
    images: List[UploadFile] = File(...),
):
    if not consent:
        raise HTTPException(status_code=400, detail="Biometric consent is required.")
    
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be blank.")

    min_img = config.enrollment.min_images
    max_img = config.enrollment.max_images
    if not (min_img <= len(images) <= max_img):
        raise HTTPException(
            status_code=400,
            detail=f"Please upload between {min_img} and {max_img} images (received {len(images)}).",
        )

    try:
        engine = get_recognition_engine()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model engine failed to load: {exc}")

    extracted_embeddings = []
    processed_images = []

    for idx, file in enumerate(images):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail=f"Image #{idx+1} could not be decoded.")

        try:
            face = face_engine.extract_single_face(img_bgr)
            extracted_embeddings.append(face.embedding)
            processed_images.append(img_bgr)
        except ValueError as val_err:
            raise HTTPException(
                status_code=400,
                detail=f"Image #{idx+1} ({file.filename}) invalid: {val_err}",
            )

    # Save images and persist
    identity_id = database.identity_id_for(clean_name)
    target_dir = person_image_dir(config.images_root, identity_id)

    rep_filename = None
    for idx, img in enumerate(processed_images):
        filename = f"img_{int(time.time())}_{idx}.jpg"
        save_path = target_dir / filename
        cv2.imwrite(str(save_path), img)
        if idx == 0:
            rep_filename = filename

    database.add_embeddings(
        name=clean_name,
        embeddings=extracted_embeddings,
        image_paths_added=len(processed_images),
        representative_image=rep_filename,
        overwrite=True,
        identity_id=identity_id,
    )

    logger.info("Successfully enrolled '%s' via web API (%d images)", clean_name, len(processed_images))
    return {
        "success": True,
        "name": clean_name,
        "identity_id": identity_id,
        "images_enrolled": len(processed_images),
    }


@app.delete("/api/persons/{name}")
async def delete_person(name: str):
    if not database.person_exists(name):
        raise HTTPException(status_code=404, detail=f"Person '{name}' does not exist.")

    identity_id = database.identity_id_for(name)
    database.delete_person(name)
    delete_person_directory(config.images_root, identity_id)
    return {"success": True, "message": f"Person '{name}' deleted successfully."}


class RecognizeRequest(BaseModel):
    image_base64: str
    target_name: Optional[str] = None


@app.post("/api/recognize")
async def recognize_frame(payload: RecognizeRequest):
    try:
        raw_data = payload.image_base64
        if "," in raw_data:
            raw_data = raw_data.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image frame.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")

    engine = get_recognition_engine()
    results = engine.recognize_frame(frame_bgr)

    detections = [res.bbox for _, res in results]
    tracker.update(detections)

    frame_height, frame_width = frame_bgr.shape[:2]
    faces_data = []

    for face, res in results:
        track_id = tracker.find_track_id_for_bbox(res.bbox)
        stable_name = res.name
        if track_id != -1:
            smoother = tracker.get_smoother(track_id)
            stable_name = smoother.update(res.name)

        p_width = bbox_pixel_width(res.bbox)
        dist_cm = estimate_distance_cm(p_width, config)
        zone_info = classify_distance(dist_cm, config)

        x1, y1, x2, y2 = res.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        faces_data.append({
            "name": stable_name,
            "raw_name": res.name,
            "is_known": res.is_known,
            "similarity": round(float(res.similarity), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int(center_x), int(center_y)],
            "track_id": track_id,
            "distance_cm": round(float(dist_cm), 1),
            "distance_zone": zone_info.zone.name,
            "distance_label": zone_info.label,
            "color_hex": zone_info.color_hex,
        })

    # Robot command calculation
    robot_data = {
        "command": RobotCommand.STOP.value,
        "reason": "No target selected",
        "target_found": False,
        "offset_px": 0,
        "distance_cm": -1.0,
    }

    if payload.target_name:
        target_face = select_recognized_target(results, payload.target_name)
        if target_face is not None:
            t_bbox = target_face.bbox
            t_center_x = (t_bbox[0] + t_bbox[2]) // 2
            t_width = bbox_pixel_width(t_bbox)
            t_dist = estimate_distance_cm(t_width, config)
            decision = decide_command(frame_width, t_center_x, t_dist, config)
            offset_px = int(t_center_x - frame_width // 2)
            robot_data = {
                "command": decision.command.value,
                "reason": decision.reason,
                "target_found": True,
                "offset_px": offset_px,
                "distance_cm": round(float(t_dist), 1),
                "target_bbox": [int(c) for c in t_bbox],
            }
        else:
            robot_data["reason"] = f"Target '{payload.target_name}' not detected in frame."

    return {
        "faces": faces_data,
        "robot": robot_data,
        "frame_size": [frame_width, frame_height],
    }


# ----------------------------------------------------------------------
# Real-Time WebSocket Endpoint
# ----------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected to live recognition stream.")
    engine = get_recognition_engine()

    try:
        while True:
            data = await websocket.receive_json()
            raw_base64 = data.get("image", "")
            target_name = data.get("target_name", None)

            if not raw_base64:
                await websocket.send_json({"error": "No image data received."})
                continue

            if "," in raw_base64:
                raw_base64 = raw_base64.split(",", 1)[1]

            img_bytes = base64.b64decode(raw_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send_json({"error": "Failed to decode frame."})
                continue

            t0 = time.time()
            results = engine.recognize_frame(frame_bgr)
            inference_time_ms = round((time.time() - t0) * 1000, 1)

            detections = [res.bbox for _, res in results]
            tracker.update(detections)
            frame_height, frame_width = frame_bgr.shape[:2]

            faces_data = []
            for face, res in results:
                track_id = tracker.find_track_id_for_bbox(res.bbox)
                stable_name = res.name
                if track_id != -1:
                    smoother = tracker.get_smoother(track_id)
                    stable_name = smoother.update(res.name)

                p_width = bbox_pixel_width(res.bbox)
                dist_cm = estimate_distance_cm(p_width, config)
                zone_info = classify_distance(dist_cm, config)
                x1, y1, x2, y2 = res.bbox

                faces_data.append({
                    "name": stable_name,
                    "is_known": res.is_known,
                    "similarity": round(float(res.similarity), 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "distance_cm": round(float(dist_cm), 1),
                    "distance_zone": zone_info.zone.name,
                    "distance_label": zone_info.label,
                    "color_hex": zone_info.color_hex,
                })

            robot_data = {
                "command": RobotCommand.STOP.value,
                "reason": "No target selected",
                "target_found": False,
                "offset_px": 0,
                "distance_cm": -1.0,
            }

            if target_name:
                target_face = select_recognized_target(results, target_name)
                if target_face is not None:
                    t_bbox = target_face.bbox
                    t_center_x = (t_bbox[0] + t_bbox[2]) // 2
                    t_width = bbox_pixel_width(t_bbox)
                    t_dist = estimate_distance_cm(t_width, config)
                    decision = decide_command(frame_width, t_center_x, t_dist, config)
                    offset_px = int(t_center_x - frame_width // 2)
                    robot_data = {
                        "command": decision.command.value,
                        "reason": decision.reason,
                        "target_found": True,
                        "offset_px": offset_px,
                        "distance_cm": round(float(t_dist), 1),
                    }
                else:
                    robot_data["reason"] = f"Target '{target_name}' not detected."

            await websocket.send_json({
                "faces": faces_data,
                "robot": robot_data,
                "inference_time_ms": inference_time_ms,
                "frame_size": [frame_width, frame_height],
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket streaming error: %s", exc)
