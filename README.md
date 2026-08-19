# VisionGuard AI — Smart Face Recognition and Robot Tracking System

[![CI](https://github.com/Priyam-Thakur/visionguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyam-Thakur/visionguard-ai/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![Computer Vision](https://img.shields.io/badge/Model-InsightFace%20Buffalo__L-orange.svg)](https://github.com/deepinsight/insightface)
[![Status](https://img.shields.io/badge/Project%20Status-Portfolio%20Prototype-green.svg)](#limitations)

**VisionGuard AI** is a modular Python desktop application that integrates real-time face detection, deep face recognition, monocular distance estimation, and autonomous target-following robot simulation (with an optional serial microcontroller interface).

Designed as an advanced computer vision portfolio project, it demonstrates clean desktop software engineering, multithreaded inference pipelines, responsible biometric data handling, and fail-safe physical control systems.

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Architecture Diagram](#architecture-diagram)
4. [Application Workflow](#application-workflow)
5. [Recognition Pipeline](#recognition-pipeline)
6. [Robot Control Pipeline](#robot-control-pipeline)
7. [Safety & Fail-Safe Design](#safety--fail-safe-design)
8. [Technology Stack](#technology-stack)
9. [Repository Structure](#repository-structure)
10. [Installation & Setup](#installation--setup)
11. [Running the Application](#running-the-application)
12. [Enrollment Workflow](#enrollment-workflow)
13. [Live Recognition Workflow](#live-recognition-workflow)
14. [Robot Simulation Mode](#robot-simulation-mode)
15. [Optional Hardware Mode](#optional-hardware-mode)
16. [Microcontroller (Arduino/ESP32) Protocol](#microcontroller-arduinoesp32-protocol)
17. [Configuration Reference](#configuration-reference)
18. [Testing Suite](#testing-suite)
19. [Continuous Integration (CI)](#continuous-integration-ci)
20. [Troubleshooting](#troubleshooting)
21. [Privacy, Data Governance & Ethics](#privacy-data-governance--ethics)
22. [Honest Engineering Limitations](#honest-engineering-limitations)
23. [Future Roadmap](#future-roadmap)
24. [Attribution & Provenance](#attribution--provenance)
25. [License](#license)

---

## Key Features

- **Decoupled Asynchronous Inference**: UI redrawing runs independently on the main event loop while InsightFace deep inference runs on a dedicated, single-slot latest-frame worker thread, preventing frame backpressure and UI freezing.
- **Robust Multi-Image Enrollment**: Enrolls identities with 2–20 uncompressed face images. Generates 512-D ArcFace feature vectors without lossy embedding averaging, maintaining granular sample galleries.
- **Secure Identity Storage**: Decouples human display names from filesystem paths using deterministic UUIDs (`data/images/<uuid>/`), preventing directory traversal and name-collision bugs.
- **Monocular Distance Estimation**: Calculates geometric target distance using pinhole camera triangle similarity principles calibrated to standard facial biometrics.
- **Temporal Label Smoothing**: Centroid-based bounding-box tracker paired with a rolling majority-vote filter eliminates single-frame recognition flickering.
- **Deterministic Target Following**: Robot mode strictly follows an explicitly selected, recognized person. Unknown or non-selected faces are strictly locked out from commanding movement.
- **Fail-Safe Safety Engine**: Proactive `STOP` dispatch on camera stream failure, recognition loss, serial interruption, window close, or explicit user Emergency Stop.
- **Integrated Hardware Simulation**: Full simulation mode allows complete algorithmic validation without requiring connected Arduino or motor hardware.

---

## System Architecture

VisionGuard AI follows a layered modular architecture with strict separation of concerns:

- **Presentation Layer (`app/ui/`)**: Built on CustomTkinter. Houses the dashboard, enrollment forms, live camera viewports, robot control panels, and configuration settings.
- **Worker & Stream Layer (`app/core/`)**:
  - `CameraStream`: Threaded OpenCV video capture with frame-drop prevention and consecutive failure counters.
  - `LatestFrameRecognitionWorker`: Bounded queue worker that pulls frames and executes inference asynchronously.
- **Inference Layer (`app/core/face_utils.py`, `embedding_utils.py`, `recognition_utils.py`)**:
  - **InsightFace Buffalo_L**: Pretrained RetinaFace face detector and ArcFace feature extractor.
  - **Cosine Metric Matcher**: Vectorized similarity comparison against stacked gallery embeddings.
- **State & Tracking Layer (`app/core/tracking_utils.py`)**:
  - Centroid nearest-neighbor tracking across frame sequences.
  - `LabelSmoother` rolling window to filter transient detection noise.
- **Safety & Control Layer (`app/core/robot_utils.py`)**:
  - Policy engine mapping horizontal pixel offset and distance bands to discrete commands (`LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, `STOP`).
  - Serial transport manager with command deduplication, fail-safe disconnect, and optional bidirectional ACK verification.
- **Persistence Layer (`app/core/database_utils.py`, `file_utils.py`)**:
  - Atomic dual-file storage: Joblib-serialized NumPy embedding arrays (`embeddings.pkl`) and human-readable metadata (`metadata.json`).

---

## Architecture Diagram

```text
                  +-----------------------------------+
                  |        Webcam Video Stream        |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Threaded CameraStream (OpenCV) |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            | (Raw Frame)                                   | (Raw Frame)
            v                                               v
+-----------------------+                       +-----------------------+
|  CustomTkinter UI     |                       | Latest-Frame Worker   |
|  (Smooth Redraw Loop) |                       | (Asynchronous Queue)  |
+-----------------------+                       +-----------------------+
            ^                                               |
            |                                               v
            |                                   +-----------------------+
            |                                   |  InsightFace Buffalo_L|
            |                                   |  - RetinaFace Detect  |
            |                                   |  - 5-Point Alignment  |
            |                                   |  - 512-D Embedding    |
            |                                   +-----------------------+
            |                                               |
            |                                               v
            |                                   +-----------------------+
            |                                   | Recognition Engine    |
            |                                   | - Cosine Similarity   |
            |                                   | - Threshold Matching  |
            |                                   | - Label Smoother      |
            |                                   +-----------------------+
            |                                               |
            +-----------------------------------------------+
            | (Annotated Frame, Tracked BBoxes, Identities)
            v
+-----------------------------------------------------------------------+
|                         Target Selector                               |
| (Only Follows Explicitly Selected, Confirmed Known Identity)          |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    Distance & Geometry Estimator                      |
|       d = (known_width_cm * focal_length_px) / face_pixel_width       |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                        Safety Controller                              |
|       - Dead-Zone Alignment (LEFT / RIGHT)                            |
|       - Range Regulation (FORWARD / BACKWARD / STOP)                  |
|       - Fail-Safe Invariants (Target Loss / Unsafe State -> STOP)     |
+-----------------------------------------------------------------------+
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
       +---------------------------+ +---------------------------+
       |   Simulated Robot Panel   | |  Microcontroller Serial   |
       |  (Zero Hardware Required) | |  (Arduino / ESP32 + ACK)  |
       +---------------------------+ +---------------------------+
```

---

## Application Workflow

1. **Initialization**: `main.py` loads and validates `config.yaml`, initializes rotating logging to `data/logs/visionguard.log`, and launches `MainWindow`.
2. **Dashboard Overview**: Displays real-time database counts, camera readiness, robot controller status, and an append-only event log.
3. **Enrollment**: Administrator enters a person’s name, reviews the biometric consent disclaimer, and uploads 2–20 clear face photos. The system detects single faces, extracts 512-D ArcFace embeddings, creates a dedicated UUID folder, and saves metadata.
4. **Live Recognition**: Starts camera stream and inference worker. Annotates faces with identity labels, similarity percentages, and real-time estimated distance.
5. **Robot Mode**: Prompts the user to select an enrolled target identity. Evaluates target position and emits movement commands to the simulation UI or over serial.

---

## Recognition Pipeline

1. **Face Detection**: Input BGR frame is processed by RetinaFace to generate bounding boxes $(x_1, y_1, x_2, y_2)$ and confidence scores ($c \ge 0.5$).
2. **Alignment & Landmark Extraction**: Five facial landmarks (eyes, nose, mouth corners) are detected to align the face crop.
3. **Feature Extraction**: ArcFace backbone produces a 512-dimensional floating-point feature representation vector $v$.
4. **L2 Normalization**: Embedding vectors are projected onto the unit hypersphere:
   $$\hat{v} = \frac{v}{\|v\|_2}$$
5. **Gallery Matching**: Query vector is matched against all enrolled embeddings via batch dot product:
   $$\text{sim}(\hat{q}, \hat{g}_i) = \hat{q} \cdot \hat{g}_i$$
   If $\max(\text{sim}) \ge \theta_{\text{thresh}}$ (default $0.45$), the face is classified as the corresponding identity; otherwise, it is marked **Unknown**.
6. **Temporal Stabilization**: A rolling majority-vote filter smooths classifications over consecutive frames to avoid flicker.

---

## Robot Control Pipeline

Movement decisions prioritize horizontal alignment first, followed by range regulation:

```text
                    Target Bounding Box Center (X)
                                  |
            +---------------------+---------------------+
            |                                           |
    |X - Center| > 60px                         |X - Center| <= 60px
            |                                           |
     [Turn Command]                         [Distance Regulation]
   X < Center -> LEFT                        Distance > 100cm -> FORWARD
   X > Center -> RIGHT                       Distance < 50cm  -> BACKWARD
                                             50cm-100cm       -> STOP (Hold)
```

---

## Safety & Fail-Safe Design

Physical robot movement can present real-world risks. VisionGuard AI enforces seven strict safety invariants:

| # | Invariant / Condition | Behavior |
| :--- | :--- | :--- |
| **1** | **Unknown Identity Detected** | **Never commands movement.** Unknown faces cannot command the robot under any circumstances. |
| **2** | **Target Disappearance** | If the selected target identity leaves the camera frame, the controller immediately dispatches `STOP`. |
| **3** | **Emergency Stop Button** | Prominent red Emergency Stop button instantly dispatches `STOP` and terminates the active session. |
| **4** | **Camera Stream Failure** | If 10 consecutive frame capture attempts fail, the stream halts and dispatches `STOP`. |
| **5** | **Inference Worker Crash** | Any unhandled background exception in the inference thread halts the robot and triggers an alert. |
| **6** | **Serial Disconnect / Shutdown** | Controller unconditionally writes `STOP\n` to the serial line prior to closing the port or exiting. |
| **7** | **Watchdog Command Timeout** | The microcontroller firmware halts motors automatically if no valid command is received within $750\,\text{ms}$. |

---

## Technology Stack

- **Core Language**: Python 3.10 / 3.11
- **Graphical Interface**: CustomTkinter 5.2.2 (Modern dark-themed Tkinter framework)
- **Computer Vision & Image I/O**: OpenCV (`opencv-python`), Pillow
- **Deep Learning Inference**: InsightFace (Buffalo_L model pack), ONNX Runtime
- **Scientific Computing**: NumPy, Scipy, Scikit-Learn
- **Serial Communications**: PySerial
- **Serialization & Config**: PyYAML, Joblib, JSON
- **Testing & Quality Assurance**: Pytest, Pytest-Cov

---

## Repository Structure

```text
visionguard-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated multi-OS GitHub Actions workflow
├── app/
│   ├── __init__.py
│   ├── config.py                  # Strongly-typed dataclass configuration
│   ├── logger.py                  # Rotating file logging configuration
│   ├── core/
│   │   ├── __init__.py
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
│   └── ui/
│       ├── __init__.py
│       ├── context.py             # Shared AppContext singleton holder
│       ├── main_window.py         # Top-level window, router & safe exit handling
│       ├── sidebar.py             # Navigation sidebar component
│       ├── theme.py               # Centralized dark-mode visual design tokens
│       ├── pages/
│       │   ├── __init__.py
│       │   ├── dashboard.py       # Metrics overview & activity log
│       │   ├── enroll_person.py   # Identity enrollment & consent verification
│       │   ├── live_recognition.py# Live webcam recognition viewport
│       │   ├── registered_persons.py # Searchable identity manager (View/Update/Delete)
│       │   ├── robot_mode.py      # Person-following robot control & simulation
│       │   └── settings.py        # Live parameter configuration form
│       └── widgets/
│           ├── __init__.py
│           ├── dialogs.py         # Progress, confirmation & conflict modal dialogs
│           ├── distance_indicator.py # Circular color-coded distance indicator
│           └── info_card.py       # Interactive dashboard summary metric card
├── data/                          # Local data directory (biometrics git-ignored)
│   ├── database/                  # embeddings.pkl, metadata.json (.gitkeep)
│   ├── images/                    # UUID image galleries (.gitkeep)
│   ├── logs/                      # visionguard.log (.gitkeep)
│   └── models/                    # InsightFace Buffalo_L ONNX weights (.gitkeep)
├── firmware/
│   └── visionguard_robot/
│       └── visionguard_robot.ino  # Reference Arduino firmware with watchdog & ACK
├── tests/
│   └── test_core.py               # Comprehensive pytest test suite (34 unit tests)
├── .gitignore                     # Strict privacy & cache exclusions
├── config.yaml                    # Single source of truth configuration file
├── main.py                        # Application entry point
├── README.md                      # Project documentation
├── requirements.txt               # Production runtime dependencies
└── requirements-dev.txt           # Test & development dependencies
```

---

## Installation & Setup

### Prerequisites
- Operating System: Windows 10/11, macOS, or Linux
- Python: Version 3.10 or 3.11 installed
- Standard USB Webcam

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Priyam-Thakur/visionguard-ai.git
   cd visionguard-ai
   ```

2. **Create and Activate a Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

*(Note: On initial execution of face recognition, InsightFace will automatically download the pretrained `buffalo_l` ONNX model pack to `data/models/`.)*

---

## Running the Application

Launch the desktop interface:
```bash
python main.py
```

---

## Enrollment Workflow

1. Navigate to **Enroll Person** in the left sidebar.
2. Enter the person's full display name (e.g., `Priyam Thakur`).
3. Review and check the **biometric consent confirmation checkbox**.
4. Click **📁 Upload Images** and select between 2 and 20 clear JPEG/PNG images of the person.
5. Click **✅ Enroll Person**.
   - Each image is evaluated to ensure it contains **exactly one** face.
   - The face crop is aligned, normalized, and converted into a 512-D vector.
   - The images are copied to `data/images/<uuid>/` and the database is updated atomically.

---

## Live Recognition Workflow

1. Navigate to **Live Recognition**.
2. Click **▶ Start Camera**.
3. View the live video feed:
   - Green bounding boxes indicate confirmed recognized identities with confidence score.
   - Red bounding boxes indicate **Unknown** persons.
   - The on-screen HUD displays camera FPS, inference FPS, and monocular distance.
4. Click **■ Stop Camera** to release the video capture device cleanly.

---

## Robot Simulation Mode

By default, `robot.simulate_hardware: true` is enabled in `config.yaml`.

1. Navigate to **Robot Mode**.
2. Select an enrolled target from the dropdown menu (e.g., `Priyam Thakur`).
3. Click **▶ Start Robot Mode**.
4. The live viewport highlights the target in amber, computes the center-offset and distance, and displays real-time commands:
   - `⬅️ LEFT` / `➡️ RIGHT` when the target moves off-center.
   - `⬆️ FORWARD` when the target is farther than $100\,\text{cm}$.
   - `⬇️ BACKWARD` when the target approaches closer than $50\,\text{cm}$.
   - `⏹️ STOP` when centered within the ideal range ($50\text{–}100\,\text{cm}$).
5. Test the **EMERGENCY STOP** button to verify immediate movement cancellation.

---

## Optional Hardware Mode

To connect to a physical robot over serial:

1. Connect your microcontroller (Arduino Uno/Nano, ESP32, STM32) via USB.
2. Identify the serial COM port (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux).
3. In **Settings**, disable *Simulate Hardware*, set the port and baud rate ($9600$), and click **💾 Save Settings**.
4. Flash the reference sketch from `firmware/visionguard_robot/visionguard_robot.ino`.

---

## Microcontroller (Arduino/ESP32) Protocol

VisionGuard AI communicates over standard UART serial using line-delimited ASCII strings terminated by `\n`:

```text
Host (VisionGuard AI)                 Microcontroller (Arduino/ESP32)
         |                                           |
         | -------- FORWARD\n ---------------------> | [Executes Forward Drive]
         | <------- ACK FORWARD\n [Optional] ------- | [Resets Watchdog Timer]
         |                                           |
         | -------- STOP\n ------------------------> | [Halts All Motors]
         | <------- ACK STOP\n [Optional] ---------- |
         |                                           |
         | [Host Disconnect / Inactive]              |
         | (No command for > 750ms)                  | [Watchdog Timer Expires]
         |                                           | [Auto-Executes STOP]
```

---

## Configuration Reference

All application parameters are consolidated in `config.yaml`:

```yaml
app:
  name: VisionGuard AI
  version: 1.0.0
  window_width: 1200
  window_height: 750
  theme: dark

paths:
  database_dir: data/database
  embeddings_file: data/database/embeddings.pkl
  metadata_file: data/database/metadata.json
  images_dir: data/images
  models_dir: data/models
  logs_dir: data/logs

face_analysis:
  model_name: buffalo_l
  providers: [CPUExecutionProvider]
  detection_size: [640, 640]
  min_face_confidence: 0.5

enrollment:
  min_images: 2
  recommended_images: 8
  max_images: 20

recognition:
  similarity_threshold: 0.45
  recognition_smoothing_window: 5
  unknown_label: Unknown

distance:
  known_face_width_cm: 14.0
  focal_length_px: 615.0
  too_close_max_cm: 50
  ideal_min_cm: 50
  ideal_max_cm: 100
  too_far_max_cm: 150

camera:
  device_index: 0
  frame_width: 960
  frame_height: 540
  target_fps: 30

robot:
  simulate_hardware: true
  serial_port: COM3
  baud_rate: 9600
  center_dead_zone_px: 60
  forward_distance_cm: 100
  backward_distance_cm: 50
  command_timeout_ms: 750
  require_ack: false
  ack_timeout_ms: 250

logging:
  level: INFO
  max_bytes: 1048576
  backup_count: 5
  log_filename: visionguard.log
```

---

## Testing Suite

VisionGuard AI features a deterministic test suite that executes completely without webcams, physical robots, or downloaded models.

Run unit tests:
```bash
pytest -v
```

Run test suite with code coverage:
```bash
pytest --cov=app tests/
```

Verify syntax and compilation:
```bash
python -m compileall app main.py tests
```

---

## Continuous Integration (CI)

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push and pull request against the `main` branch. It executes:
- Matrix builds across **Python 3.10 and 3.11** on both **Ubuntu** and **Windows**.
- Dependency resolution from `requirements-dev.txt`.
- Headless compilation validation.
- The 34-case automated Pytest suite.

---

## Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Model Load Error** | Initial InsightFace model download interrupted or missing ONNX Runtime. | Ensure an active internet connection on the first run; verify `onnxruntime` is installed. |
| **Camera Unavailable** | Webcam index incorrect or locked by another application (Zoom, Teams, Browser). | Close competing camera apps. Test changing `camera.device_index` to `1` or `2` in Settings. |
| **Serial Connection Failed** | Selected COM port is invalid or robot is not connected. | Keep `simulate_hardware: true` enabled unless real hardware is connected and verified. |
| **Recognition False Negatives** | Lighting variation or similarity threshold too strict. | Ensure enrollment photos have diverse angles and lighting; adjust threshold in Settings. |

---

## Privacy, Data Governance & Ethics

- **Strictly Local Processing**: All face detections, embeddings, and image galleries are processed and stored strictly on the local machine. No images or biometric vectors are transmitted over any network.
- **Informed Consent**: The enrollment UI requires an explicit acknowledgment that biometric consent was obtained before storing identity data.
- **Complete Deletion Right**: Deleting a person from the *Registered Persons* page permanently removes their stored embeddings from `embeddings.pkl`, metadata from `metadata.json`, and deletes their image directory from disk.
- **Git Hygiene**: Strict `.gitignore` rules prevent accidental commits of real biometric images (`data/images/*`), embedding databases (`data/database/*`), or runtime logs containing personal information.

---

## Honest Engineering Limitations

In alignment with responsible engineering and academic integrity:

1. **Biometric Security**: This project is a prototype demonstration. It does **not** include active infrared liveness detection or 3D structured-light anti-spoofing; it can be fooled by high-resolution photographs or video playback.
2. **Monocular Distance**: Distance is mathematically estimated from 2D pixel widths assuming a standard adult facial width of $14\,\text{cm}$. It is sensitive to head tilt, facial expression, and focal distortion, and must **never** be used as a primary collision-avoidance sensor on heavy physical robots.
3. **No Fabricated Benchmarks**: Actual inference FPS varies based on host CPU/GPU hardware. No false accuracy claims (e.g., "99.9% real-world accuracy") are made.

---

## Future Roadmap

- [ ] Liveness detection module (blink and micro-motion texture analysis).
- [ ] Multi-camera feed support.
- [ ] TensorRT and DirectML ONNX execution provider acceleration.
- [ ] ROS2 (Robot Operating System) node integration for autonomous mobile bases.
- [ ] Interactive focal length calibration wizard tool.

---

## Attribution & Provenance

- **Maintained & Modernized by**: **Priyam Thakur**
- **Original Upstream Base**: Public computer vision foundation authored by **PRERANA P JOIS**. Modernized, refactored, hardened, and expanded with asynchronous threading, safety controllers, fail-safes, testing, and modern UI architecture.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
