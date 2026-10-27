# VisionGuard, explained simply

## One sentence

VisionGuard is face recognition that runs entirely in your browser, plus the
safety rules that let a robot follow one chosen person and ignore everyone else.

## The 30-second version (say this first)

"You open the website and point your camera. It finds every face, and you can
type a name for yours. From then on it recognises you, shows how far away you
are, and a simulated robot turns and drives to keep you in front of it.
Strangers show as Unknown and can never move the robot. The interesting part is
that the two neural networks run inside the browser with WebAssembly, so no
video ever leaves your device and the site costs nothing to host."

## How it works, step by step

1. **Find faces.** A small detector network (SCRFD, 2.5 MB) returns a box,
   a confidence score and 5 landmarks (eyes, nose, mouth corners) for each face.
2. **Line the face up.** The 5 landmarks are mapped onto fixed reference points
   with a rotation + scale + shift (a similarity transform), giving an upright
   112×112 crop. Without this, a tilted head would look like a different person.
3. **Make a faceprint.** ArcFace (MobileFaceNet, 13 MB) turns the crop into
   512 numbers, scaled to length 1. The site draws these as the heat grid.
4. **Compare.** The dot product of two faceprints is their cosine similarity.
   0.45 or more = same person. Unrelated faces score around 0.0–0.25.
5. **Track and smooth.** Faces are matched across frames by nearest centre
   point; the name shown is the majority vote of the last 5 frames, so one bad
   frame can't make the label flicker.
6. **Distance.** A face is about 14 cm wide, so with the pinhole camera model
   distance = 14 cm × focal length ÷ face width in pixels.
7. **Robot decision.** Turn left or right until the person is within ±60 px of
   centre, then move forward if they're over 100 cm away or back if under 50 cm.

## Six things that make it stand out

- **It runs in the browser.** onnxruntime-web runs the same InsightFace models
  in a Web Worker. The page stays smooth because inference is off the main thread.
- **I proved the port is correct.** I ran the JavaScript pipeline and the
  original Python InsightFace on the same photo. Same 6 faces, boxes overlapping
  ≥ 99.7%, faceprints matching at cosine 0.996–0.9996.
- **Safety first.** Unknown faces can't drive; losing the target stops the robot;
  the emergency stop overrides everything. On the real robot, the Arduino stops
  the motors if it hears nothing for 750 ms.
- **It refuses bad photos.** Before saving a face it checks four things a
  recruiter can see as green ticks: big enough, sharp (variance of the
  Laplacian, a standard blur measure), well lit, and facing the camera (from
  the landmarks). Blurry or tiny frames are skipped and it tells you what to
  fix ("Move closer", "Hold still"). It also skips near-duplicate frames, so
  the 5 saved faceprints cover slightly different angles.
- **You can see the maths.** The face map squashes the 512 numbers down to 2
  with principal component analysis, so you can literally see one person's
  faceprints clumping together and your live face landing inside the clump.
- **A real performance fix.** Re-identifying every face every frame ran at 3 fps
  with six people. Re-identifying each tracked face only every 6 frames (the
  tracker carries the name in between) brought it to about 20 fps.

## Two-minute demo script

1. Open the site: the sample photo is analysed straight away (box, landmarks,
   faceprint, timings).
2. Click **Use my camera**. Type your name and press **Remember**: the ticks
   show the quality checks, and the bar fills as it collects 5 good frames.
   Sit an arm's length away and press **Calibrate distance**.
3. Move left and right: the label follows you, and the distance updates.
4. Scroll to **Robot**: choose yourself. It says LEFT / RIGHT / FORWARD.
   Cover your face or step away: "target lost", STOP. Press **Emergency stop**.
5. Scroll to **Face map**: your saved faceprints are a cluster, and the white
   ring (you, live) sits inside it.
6. Scroll to **How it works**: every number there is from your face, live.

## Questions you'll likely get

**Why run it in the browser instead of on a server?**
The server version got killed for running out of memory on the free 512 MB
instance. In the browser it's free to host, faster (no network round trip) and
private, since video never leaves the device.

**Is it training a model when you enrol someone?**
No. The networks are pretrained and never change. Enrolling just stores
faceprints; recognition is a nearest-match search against them.

**Why 0.45 as the threshold?**
It's the project's setting for ArcFace. Higher means fewer false matches but
more misses. In the demo, different people scored at most 0.23 and the same
person close to 1.0 on the same photo; across real lighting changes the same
person typically scores 0.5–0.8.

**How accurate is the distance?**
It assumes an average face width (14 cm). Out of the box it also assumes a
typical webcam focal length, so expect roughly ±20%. Pressing **Calibrate
distance** while sitting 50 cm away measures your camera's real focal length
(focal = face width in px × 50 ÷ 14), which removes most of that error.

**How do you stop it learning from a blurry frame?**
Each aligned 112×112 crop is scored for sharpness with the variance of the
Laplacian: edges give big values, blur flattens them. I calibrated the cut-off
(60) on real crops: sharp faces scored 120–420, slightly blurred ones 15–40.
Brightness, face size and head pose (from the eye and nose landmarks) are
checked too.

**What is the face map actually showing?**
PCA finds the two directions in which the saved faceprints vary most and
projects every faceprint onto them. It's for intuition only: matching still
uses all 512 numbers, and the page shows how much of the variation the 2D
picture keeps.

**What happens with several people?**
Each face gets a track. The robot follows the largest face whose name matches
the chosen target; everyone else is ignored, and unknown people can never
trigger movement.

**What would you add next?**
Liveness detection so a printed photo can't fool it; WebGPU for faster
inference on supported browsers.

**What's the honest limitation?**
The browser uses the compact model (MobileFaceNet) for speed; the desktop app
uses the larger ResNet-50 pack, which is more accurate in difficult lighting.
The InsightFace models are licensed for non-commercial research only.

## Resume bullets

- Built **VisionGuard**, a face-recognition and person-following system: SCRFD
  detection, landmark alignment and 512-D ArcFace embeddings with cosine
  matching, centroid tracking with temporal smoothing, and a fail-safe robot
  control policy (Python desktop app + Arduino firmware with a 750 ms watchdog).
- Ported the InsightFace pipeline to the browser with **onnxruntime-web
  (WebAssembly) in a Web Worker**; verified parity with Python (detections
  ≥ 99.7% IoU, embedding cosine ≥ 0.996). This fixed out-of-memory crashes in
  the cloud demo and made it private (no uploads) and free to host.
- Added an **enrolment quality gate** (Laplacian-variance sharpness, brightness,
  size and landmark-based head pose) with live fix-it hints, one-click
  **focal-length calibration** for distance, and a **PCA "face map"** that
  visualises the 512-D embedding space in the browser.
- Raised live throughput from **3 to ~20 fps** with six faces by re-identifying
  tracked faces every 6 frames instead of every frame; 57 automated tests
  (41 Python + 16 JS) in CI.
