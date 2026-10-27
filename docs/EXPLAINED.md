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

## Four things that make it stand out

- **It runs in the browser.** onnxruntime-web runs the same InsightFace models
  in a Web Worker. The page stays smooth because inference is off the main thread.
- **I proved the port is correct.** I ran the JavaScript pipeline and the
  original Python InsightFace on the same photo. Same 6 faces, boxes overlapping
  ≥ 99.7%, faceprints matching at cosine 0.996–0.9996.
- **Safety first.** Unknown faces can't drive; losing the target stops the robot;
  the emergency stop overrides everything. On the real robot, the Arduino stops
  the motors if it hears nothing for 750 ms.
- **A real performance fix.** Re-identifying every face every frame ran at 3 fps
  with six people. Re-identifying each tracked face only every 6 frames (the
  tracker carries the name in between) brought it to about 20 fps.

## Two-minute demo script

1. Open the site: the sample photo is analysed straight away (box, landmarks,
   faceprint, timings).
2. Click **Use my camera**. Type your name and press **Remember**: it learns
   from 5 frames.
3. Move left and right: the label follows you, and the distance updates.
4. Scroll to **Robot**: choose yourself. It says LEFT / RIGHT / FORWARD.
   Cover your face or step away: "target lost", STOP. Press **Emergency stop**.
5. Scroll to **How it works**: every number there is from your face, live.

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
It's an estimate: it assumes an average face width and a typical webcam focal
length, so expect roughly ±20%. That's enough for "too close / ideal / too
far" bands, which is all the robot needs.

**What happens with several people?**
Each face gets a track. The robot follows the largest face whose name matches
the chosen target; everyone else is ignored, and unknown people can never
trigger movement.

**What would you add next?**
Liveness detection so a printed photo can't fool it; a calibration step for the
focal length; WebGPU for faster inference on supported browsers.

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
- Raised live throughput from **3 to ~20 fps** with six faces by re-identifying
  tracked faces every 6 frames instead of every frame; 53 automated tests
  (41 Python + 12 JS) in CI.
