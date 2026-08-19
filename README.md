# VisionGuard AI — Smart Face Recognition & Robot Tracking System

[![CI](https://github.com/priyamthakur275/visionguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/priyamthakur275/visionguard-ai/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20ASGI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61DAFB.svg)](https://react.dev/)
[![Model](https://img.shields.io/badge/Model-InsightFace%20Buffalo__L-orange.svg)](https://github.com/deepinsight/insightface)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionGuard AI** is an advanced computer vision and autonomous robotics tracking system available as both a **native Python desktop application** and a **cloud-ready web demo**.

It integrates deep facial feature extraction (**InsightFace ArcFace 512-D**), monocular pinhole distance estimation, centroid tracking with temporal majority-vote smoothing, and a fail-safe navigation policy engine for autonomous person following.

---

## Dual-Distribution Architecture

VisionGuard AI is architected with a strict separation between core vision logic, native hardware desktop execution, and public cloud deployment:

```text
                               +-------------------------------------------------------------+
                               |                 VisionGuard AI Core Engine                  |
                               |  - app.core.face_utils (InsightFace Buffalo_L ONNX)         |
                               |  - app.core.embedding_utils (L2 Normalization, Cosine Sim)  |
                               |  - app.core.distance_utils (Pinhole Triangle Similarity)    |
                               |  - app.core.tracking_utils (Centroid Tracker & Smoother)    |
                               |  - app.core.robot_utils (Command Policy & Fail-Safes)       |
                               |  - app.core.database_utils (Atomic Embeddings & Metadata)   |
                               +-------------------------------------------------------------+
                                              |                              |
                   +--------------------------+                              +--------------------------+
                   |                                                                                    |
                   v                                                                                    v
+------------------------------------+                                                +------------------------------------+
|    Native Desktop App (Python)     |                                                |      Web Demo Backend (FastAPI)    |
| - CustomTkinter Dark UI            |                                                | - CORS & Input Validation          |
| - Direct USB Webcam (OpenCV)       |                                                | - REST Endpoints:                  |
| - Physical PySerial to Arduino     |                                                |   /health, /api/status,            |
| - Local disk storage               |                                                |   /api/recognize, /api/enroll,     |
| - Packaged via PyInstaller (.exe)  |                                                |   /api/persons                     |
+------------------------------------+                                                | - Low-Latency WebSocket (/ws/live) |
                                                                                      +------------------------------------+
                                                                                                        |
                                                                                                        v
                                                                                      +------------------------------------+
                                                                                      |     React + Vite Web Frontend      |
                                                                                      | - Modern Dark AI Dashboard         |
                                                                                      | - Browser Webcam & Image Upload    |
                                                                                      | - Live Overlay (BBoxes, Distances) |
                                                                                      | - 2D Canvas Robot Telemetry Sim    |
                                                                                      | - Target Selector & Emergency Stop |
                                                                                      | - System Status & Architecture     |
                                                                                      | - Deployable to Vercel             |
                                                                                      +------------------------------------+
```

---

## Cloud vs. Local Hardware Boundary

| Dimension | Native Desktop Application | Cloud / Browser Web Demo |
| :--- | :--- | :--- |
| **Primary Use Case** | Local workstations & physical robot control | Public web portfolio demo & remote verification |
| **Video Input** | Direct OS USB webcam via OpenCV (`VideoCapture`) | Client-side browser webcam (`navigator.mediaDevices`) or image upload |
| **Robot Movement** | Physical UART serial to Arduino / ESP32 (`COM3`) | Interactive 2D HTML5 Canvas real-time vector simulation |
| **Inference Location** | Local Host CPU/GPU | Cloud Server CPU/GPU |
| **Data Persistence** | Local atomic files (`data/database/`, `data/images/`) | Isolated persistent storage directory / ephemeral RAM |
| **Packaging / Hosting** | Windows Standalone Executable (`.exe`) | Backend on **Render**, Frontend on **Vercel**, or **Docker** |

---

## Key Features

- **Asynchronous Inference Pipeline**: Dedicated background worker runs neural network operations asynchronously, eliminating UI freeze and frame backpressure.
- **ArcFace 512-D Biometric Embeddings**: Extracts high-dimensional normalized feature vectors without lossy average pooling, maintaining multi-sample galleries per identity.
- **Monocular Pinhole Distance Estimation**: Calculates geometric target distance in real-time ($d = \frac{W_{\text{known}} \cdot f_{\text{px}}}{w_{\text{px}}}$) based on calibrated facial proportions.
- **Spatial Centroid Tracking & Label Smoothing**: Nearest-neighbor Euclidean tracker coupled with a rolling window majority-vote filter eliminates single-frame detection flickering.
- **Deterministic Autonomous Following**: Robot navigation policy commands `LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, and `STOP` based on horizontal center offset ($\pm 60\,\text{px}$) and range bands ($50\text{–}100\,\text{cm}$).
- **Seven Fail-Safe Invariants**: Strict safety interlocks ensuring unknown identities never command movement, target disappearance halts immediately, and serial watchdog halts motors after $750\,\text{ms}$.

---

## System Architecture Pipeline

```text
[ Camera Frame / Upload ]
          │
          ▼
[ RetinaFace Face Detection ] ──► Extracts Bounding Box (x1,y1,x2,y2) & 5 Landmarks
          │
          ▼
[ ArcFace 512-D Embedding ] ───► L2 Normalization onto Unit Hypersphere (||v|| = 1.0)
          │
          ▼
[ Cosine Similarity Matcher ] ─► Vectorized dot-product against enrolled gallery (θ = 0.45)
          │
          ▼
[ Centroid Tracker & Smoother] ─► Spatial track association + temporal majority voting
          │
          ▼
[ Geometric Distance Estimator] ─► d = (14.0cm * 615px) / face_pixel_width
          │
          ▼
[ Safety & Command Controller ] ──► Dead-zone alignment (±60px) & Range regulation (50-100cm)
          │
          ├───────────────────────────────┐
          ▼                               ▼
[ Physical Arduino Microcontroller ]   [ 2D Interactive Web Canvas ]
  (Line-delimited UART + 750ms Watchdog)  (Real-time Heading & Tracking Ray)
```

---

## Safety & Fail-Safe Invariants

| # | Fail-Safe Mechanism | Behavior |
| :--- | :--- | :--- |
| **1** | **Unknown Identity Lockout** | Unknown faces are **strictly prohibited** from issuing movement commands under all conditions. |
| **2** | **Target Loss Safe Halt** | If the designated target identity leaves the field of view, the controller dispatches `STOP` immediately. |
| **3** | **Emergency Stop Override** | Dedicated software and hardware Emergency Stop overrides all movement commands unconditionally. |
| **4** | **Camera Stream Monitor** | Halts navigation and stream if 10 consecutive capture attempts fail or hardware is unplugged. |
| **5** | **Worker Exception Isolation** | Any inference crash safely dispatches `STOP` and displays a non-fatal recovery dialog. |
| **6** | **Disconnect Safe Shutdown** | Transmits `STOP\n` over serial prior to closing the port or terminating the process. |
| **7** | **750ms Microcontroller Watchdog** | Arduino firmware automatically cuts motor power if no valid command arrives within $750\,\text{ms}$. |

---

## Quick Start & Local Running

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+ & npm (for frontend development)
- USB Webcam (for native desktop mode or browser live streaming)

---

### 1. Run the Native Desktop Application (CustomTkinter)
```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # macOS/Linux

# Run application
python main.py
```

---

### 2. Run the Full-Stack Web Demo Locally

#### Terminal 1 — Start FastAPI Backend:
```bash
uvicorn app.web.server:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — Start React + Vite Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

### 3. Run with Docker
```bash
docker-compose up --build
```
Open **`http://localhost:8000`** in your browser.

---

### 4. Build Standalone Windows Executable (.exe)
```bash
python scripts/build_windows.py
```
Packaged output will be generated in `dist/VisionGuardAI/`.

---

## REST & WebSocket API Reference

The FastAPI backend exposes the following production endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` or `/api/health` | Service health status, model loaded state, total identities. |
| `GET` | `/api/status` or `/api/stats` | Detailed telemetry (embeddings count, threshold, distance bands). |
| `GET` | `/api/persons` | List registered identities with metadata (UUID, photo count, dates). |
| `POST` | `/api/enroll` | Multipart form (`name`, `consent`, `images[]`) to enroll a new identity. |
| `POST` | `/api/recognize` | Single-frame JSON request (`image_base64`, `target_name`) returning bounding boxes and robot decisions. |
| `DELETE`| `/api/persons/{name}` | Deletes person metadata, embeddings, and image directory. |
| `WS` | `/ws/live` | Sub-50ms bidirectional live video stream pipeline. |

---

## Automated Test Suite

VisionGuard AI features a deterministic test suite (41 tests) that executes without physical hardware or model downloads:

```bash
# Run all tests
python -m pytest -v

# Run with test coverage
python -m pytest --cov=app tests/
```

### Test Coverage Highlights
- **Config & Invariants**: 99%
- **Distance & Embedding Math**: 100%
- **Centroid Tracking & Smoothing**: 96%
- **Database & UUID File Storage**: 86%
- **Web REST & WebSocket APIs**: 100%

---

## Cloud Deployment Guide

### Deploying Backend to Render
1. Create a **Web Service** on [Render.com](https://render.com).
2. Connect your GitHub repository `https://github.com/priyamthakur275/visionguard-ai`.
3. Set **Runtime** to `Python 3` (or use `render.yaml`).
4. **Build Command**: `pip install -r requirements-dev.txt`
5. **Start Command**: `uvicorn app.web.server:app --host 0.0.0.0 --port $PORT`
6. **Health Check Path**: `/health`

### Deploying Frontend to Vercel
1. Import the repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. **Framework Preset**: `Vite`.
4. **Environment Variables**: Set `VITE_API_URL` to your Render backend URL (e.g. `https://visionguard-api.onrender.com`).
5. Deploy.

---

## Repository Structure

```text
visionguard-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated multi-OS GitHub Actions CI
├── app/
│   ├── config.py                  # Strongly-typed dataclass configuration
│   ├── logger.py                  # Rotating file logging configuration
│   ├── core/
│   │   ├── camera_utils.py        # Threaded camera stream with failure detection
│   │   ├── database_utils.py      # Face gallery persistence & metadata sync
│   │   ├── distance_utils.py      # Monocular distance estimation & zone classification
│   │   ├── embedding_utils.py     # L2 normalization & vectorized cosine similarity
│   │   ├── face_utils.py          # InsightFace Buffalo_L wrapper
│   │   ├── file_utils.py          # Safe UUID filesystem helpers
│   │   ├── image_utils.py         # Image decoding, color conversion & resizing
│   │   ├── recognition_utils.py   # Recognition engine & label smoothing
│   │   ├── recognition_worker.py  # Asynchronous latest-frame inference worker
│   │   ├── robot_utils.py         # Robot command decision logic & serial controller
│   │   └── tracking_utils.py      # Centroid tracker with grace-period aging
│   ├── ui/
│   │   ├── main_window.py         # Top-level window & navigation router
│   │   ├── pages/                 # Dashboard, Enroll, Live, Persons, Robot, Settings
│   │   └── widgets/               # Dialogs, distance indicator, info cards
│   └── web/
│       ├── server.py              # FastAPI backend API & WebSocket streamer
│       └── static/                # Static fallback assets
├── frontend/                      # React 18 + Vite Web Application
│   ├── src/
│   │   ├── components/            # Hero, LiveDemo, RobotSimulator, Architecture, Safety
│   │   ├── App.jsx                # Main single-page application layout
│   │   └── index.css              # Dark AI engineering design system
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json                # Vercel SPA routing configuration
├── firmware/
│   └── visionguard_robot/
│       └── visionguard_robot.ino  # Reference Arduino firmware with 750ms watchdog
├── scripts/
│   └── build_windows.py           # Standalone Windows packaging script
├── tests/
│   ├── test_core.py               # Core desktop unit tests (34 tests)
│   └── test_web.py                # Web backend integration tests (7 tests)
├── Dockerfile                     # Multi-stage Linux container
├── docker-compose.yml             # Local orchestration
├── render.yaml                    # Render.com deployment manifest
├── visionguard.spec               # PyInstaller specification
└── config.yaml                    # Central application configuration
```

---

## Privacy, Data Governance & Ethics

- **Zero Unconsented Storage**: During live demo streaming, frames are processed strictly in server RAM and immediately discarded.
- **Informed Consent**: Enrollment requires explicit confirmation that biometric consent was obtained.
- **Permanent Deletion**: Deleting an identity from the gallery permanently purges all stored embeddings and image folders.
- **Git Hygiene**: Strict `.gitignore` rules prevent accidental commits of real biometric images (`data/images/*`), database vectors (`data/database/*`), or runtime logs.

---

## Honest Engineering Limitations

1. **Biometric Security Prototype**: This application does not include active infrared 3D structured-light anti-spoofing; high-resolution photos can fool 2D face recognition.
2. **Monocular Geometry**: Distance estimation assumes standard adult facial dimensions ($14\,\text{cm}$). Head tilt and facial orientation affect pixel width readings.
3. **Zero Fabricated Claims**: Accuracy numbers and inference FPS reflect actual hardware benchmarks on host hardware without inflated statistics.

---

## Attribution & Provenance

- **Created, maintained, modernized, hardened, and extended by Priyam Thakur**
- VisionGuard AI is a smart face-recognition and robot-tracking system featuring asynchronous processing, safety controls, fail-safe mechanisms, automated testing, and a modern UI architecture.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.