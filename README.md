# VisionGuard AI: face recognition and robot following

[![CI](https://github.com/priyamthakur275/visionguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/priyamthakur275/visionguard-ai/actions)
[![Model](https://img.shields.io/badge/Model-InsightFace%20ArcFace-orange.svg)](https://github.com/deepinsight/insightface)
[![Runtime](https://img.shields.io/badge/Web-onnxruntime--web%20(WebAssembly)-6E4AFF.svg)](https://onnxruntime.ai/docs/tutorials/web/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live demo:** https://visionguard-ai-theta.vercel.app

VisionGuard finds every face in a camera frame, recognises the people you've
enrolled, estimates how far away they are, and steers a robot towards one of
them while never letting an unknown face move it.

The same core runs in three places:

| | Website | Desktop app | Robot |
|---|---|---|---|
| Runs on | Your browser (WebAssembly, in a Web Worker) | Windows/macOS/Linux (Python) | Arduino / ESP32 |
| Models | InsightFace **buffalo_sc**: SCRFD-500M detector + ArcFace MobileFaceNet (16 MB) | InsightFace **buffalo_l**: RetinaFace + ArcFace ResNet-50 | – |
| Camera | Browser webcam or uploaded photo | USB webcam via OpenCV | – |
| Robot output | On-screen top-down simulation | Serial commands (`LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, `STOP`) | Drives the motors, 750 ms watchdog |
| Privacy | **Nothing is uploaded.** Only 512-number faceprints are stored, in the browser | Local files | – |

---

## Why the website runs the models in the browser

The first cloud version ran InsightFace on a free 512 MB server. Loading the
model pack got the process killed for running out of memory, so the public demo
crashed on the first recognition. Moving inference to the client fixed that and
improved everything else:

- **Free to host forever**: the site is static files on Vercel; there's no server to run out of memory.
- **Real-time**: ~20 fps with six faces in view on a laptop CPU, no network round trip.
- **Private by design**: camera frames never leave the device. Faceprints are kept in `localStorage`.
- **Same maths as Python**: the JavaScript port was checked against Python InsightFace on the same photo. It found the same 6 faces (box overlap ≥ 99.7%), and its embeddings matched with cosine similarity 0.996–0.9996.

## Pipeline

```text
camera frame / photo
   │
   ▼  1. SCRFD detector (det_500m.onnx)      → boxes, scores, 5 landmarks per face
   ▼  2. Similarity-transform alignment       → upright 112×112 crop (ArcFace template)
   ▼  3. ArcFace MobileFaceNet (w600k_mbf)    → 512-D faceprint, L2-normalised
   ▼  4. Cosine similarity vs enrolled people → match if ≥ 0.45, else "Unknown"
   ▼  5. Centroid tracker + 5-frame majority vote, pinhole distance (14 cm × f ÷ width px)
   ▼  6. Robot policy + fail-safes            → LEFT / RIGHT / FORWARD / BACKWARD / STOP
```

**Real-time trick:** in live video the detector runs every frame, but a tracked
face is only re-identified when its track is new or every 6 frames. In between,
the tracker carries the faceprint forward. With six faces this took the demo
from 3 fps to about 20 fps.

## Safety and fail-safes

| # | Fail-safe | Behaviour | Where |
|---|---|---|---|
| 1 | Unknown identity lockout | Unknown faces can never issue a movement command | Web + desktop |
| 2 | Target lost, halt | If the chosen person leaves the frame, `STOP` | Web + desktop |
| 3 | Emergency stop override | E-stop overrides every rule, every frame | Web + desktop |
| 4 | No target, no motion | Nothing moves until a person is explicitly chosen | Web + desktop |
| 5 | Camera stream monitor | Halts after 10 consecutive failed captures | Desktop |
| 6 | Worker exception isolation | An inference crash sends `STOP` | Desktop |
| 7 | Safe shutdown | `STOP\n` is sent before closing the serial port | Desktop |
| 8 | 750 ms watchdog | Firmware cuts the motors if no command arrives in 750 ms | Arduino |

---

## Run it

### Website (no Python needed)
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm test           # 12 unit tests: SCRFD decoding, alignment, matching, tracker, robot fail-safes
```

### Desktop app
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Build a Windows executable with `python scripts/build_windows.py` (output in `dist/VisionGuardAI/`).

### REST / WebSocket API (optional)
```bash
uvicorn app.web.server:app --port 8000
```
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, model state, identity count |
| `GET` | `/api/stats` | Embedding count, threshold, distance bands |
| `GET` | `/api/persons` | Enrolled identities |
| `POST` | `/api/enroll` | Multipart `name`, `consent`, `images[]` |
| `POST` | `/api/recognize` | JSON `image_base64`, `target_name` → faces + robot decision |
| `DELETE` | `/api/persons/{name}` | Remove an identity |
| `WS` | `/ws/live` | Live frame stream |

Set `VISIONGUARD_MODEL=buffalo_sc` on small cloud instances (the Docker image does
this and downloads the pack at build time); the desktop default stays `buffalo_l`.

### Tests
```bash
python -m pytest -q          # 41 Python tests, no camera or model download needed
cd frontend && npm test      # 12 JavaScript tests
```
CI runs both on every push.

---

## Repository structure

```text
app/core/          Python core: face_utils (InsightFace), embedding, recognition, tracking,
                   distance, robot policy + serial controller, gallery storage
app/ui/            Desktop app (CustomTkinter)
app/web/server.py  FastAPI REST + WebSocket API
frontend/
  src/vision/      Browser engine: core.js (SCRFD decode, NMS, alignment, matching),
                   follow.js (tracker, distance, robot policy), worker.js (onnxruntime-web)
  src/components/  Viewfinder, Faceprint, RobotSection, HowItWorks
  public/models/   det_500m.onnx, w600k_mbf.onnx (InsightFace buffalo_sc)
  tests/           Node unit tests + browser parity page
firmware/          Arduino firmware with 750 ms watchdog
tests/             Python test suite
```

## Credits and licences

Code: MIT. Models: [InsightFace](https://github.com/deepinsight/insightface)
buffalo_sc / buffalo_l, released for **non-commercial research use**.
Sample photo: NASA, public domain. Distance is an estimate based on an average
face width of 14 cm.
