import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { analyze, detectFaces, embedFaces, loadEngine } from "../vision/engine.js";
import { dot, faceQuality, matchIdentity } from "../vision/core.js";
import { CONFIG, CentroidTracker, decide, distanceZone, estimateDistanceCm, setFocalLength } from "../vision/follow.js";

const VisionContext = createContext(null);
const STORE_KEY = "visionguard.gallery.v1";
const FOCAL_KEY = "visionguard.focal.v1";
const CALIBRATION_CM = 50;
const MAX_ENROL_FRAMES = 240; // give up politely after ~10-15 s of unusable frames
const SAMPLES_PER_ENROLL = 5;
const REID_EVERY = 6; // live video: refresh each tracked face's faceprint every N frames

function loadGallery() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    return Object.fromEntries(Object.entries(raw).map(([k, list]) => [k, list.map((a) => Float32Array.from(a))]));
  } catch {
    return {};
  }
}

function saveGallery(g) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(Object.fromEntries(Object.entries(g).map(([k, l]) => [k, l.map((e) => Array.from(e, (v) => +v.toFixed(5)))]))));
  } catch {
    /* storage blocked: the gallery still works for this visit */
  }
}

function loadFocal() {
  try {
    const v = Number(localStorage.getItem(FOCAL_KEY));
    return v > 100 && v < 5000 ? v : null;
  } catch {
    return null;
  }
}

export function VisionProvider({ children }) {
  const [engine, setEngine] = useState({ status: "loading", loadMs: null, error: null });
  const [mode, setMode] = useState("sample"); // sample | photo | camera
  const [photoUrl, setPhotoUrl] = useState(null);
  const [frame, setFrame] = useState(null);
  const [gallery, setGallery] = useState(loadGallery);
  const [target, setTarget] = useState("");
  const [estop, setEstop] = useState(false);
  const [selected, setSelected] = useState(0);
  const [enrolling, setEnrolling] = useState(null); // { name, got }
  const [cameraError, setCameraError] = useState(null);
  const [fps, setFps] = useState(null);
  const [focal, setFocal] = useState(loadFocal);
  const [notice, setNotice] = useState(null);

  const mediaRef = useRef(null); // current <img> or <video>
  const streamRef = useRef(null);
  const trackerRef = useRef(new CentroidTracker());
  const galleryRef = useRef(gallery);
  const enrollRef = useRef(null);
  const loopRef = useRef(0);
  galleryRef.current = gallery;

  useEffect(() => {
    loadEngine()
      .then((r) => setEngine({ status: "ready", loadMs: r.loadMs, error: null }))
      .catch((e) => setEngine({ status: "error", loadMs: null, error: e.message }));
  }, []);

  useEffect(() => saveGallery(gallery), [gallery]);
  useEffect(() => {
    setFocalLength(focal);
    try {
      focal ? localStorage.setItem(FOCAL_KEY, String(focal)) : localStorage.removeItem(FOCAL_KEY);
    } catch { /* ignore */ }
  }, [focal]);

  /** Turn raw detections into named, tracked, ranged faces. */
  const interpret = useCallback((result, { live, ids: given }) => {
    const g = galleryRef.current;
    const tracker = trackerRef.current;
    if (!live) tracker.reset();
    const ids = given ?? tracker.update(result.faces.map((f) => f.bbox));
    const faces = result.faces.map((f, i) => {
      const m = matchIdentity(f.embedding, g, CONFIG.similarityThreshold);
      const label = live ? tracker.smooth(ids[i], m.name) : m.name;
      const isKnown = label !== "Unknown";
      const distanceCm = estimateDistanceCm(f.bbox[2] - f.bbox[0], result.width);
      const check = faceQuality(f, result.width);
      return { ...f, ...m, name: label, isKnown, trackId: ids[i], distanceCm, zone: distanceZone(distanceCm), check };
    });

    // Multi-frame enrolment with a quality gate: follow the face that was named
    // (by track id; the largest face if that track is lost), accept
    // the frame only if every check passes and it isn't a near-copy of a
    // sample we already have (so the stored samples cover small pose changes).
    const job = enrollRef.current;
    if (job) {
      job.frames++;
      const biggest = faces.find((f) => f.trackId === job.trackId)
        ?? (faces.length ? faces.reduce((a, b) => (b.bbox[2] - b.bbox[0] > a.bbox[2] - a.bbox[0] ? b : a)) : null);
      if (biggest) job.trackId = biggest.trackId;
      let fix = biggest ? null : "Step into view";
      if (biggest) {
        if (!biggest.check.ok) fix = biggest.check.checks.find((c) => !c.ok).fix;
        else if (job.samples.some((s) => dot(s, biggest.embedding) > 0.985)) fix = "Turn your head slightly";
        else job.samples.push(biggest.embedding);
      }
      const done = job.samples.length >= job.needed;
      const gaveUp = !done && job.frames >= MAX_ENROL_FRAMES;
      setEnrolling(done || gaveUp ? null : { name: job.name, got: job.samples.length, fix, trackId: job.trackId });
      if (done || (gaveUp && job.samples.length >= 2)) {
        enrollRef.current = null;
        setGallery((prev) => ({ ...prev, [job.name]: [...(prev[job.name] || []), ...job.samples] }));
        setTarget((t) => t || job.name);
        setNotice(done ? `Remembered ${job.name} from ${job.samples.length} good frames.` : `Saved ${job.name} from ${job.samples.length} good frames (lighting or distance made the rest unusable).`);
      } else if (gaveUp) {
        enrollRef.current = null;
        setNotice(`Couldn't get clear frames of ${job.name}. ${fix || "Try better light, closer to the camera"}.`);
      }
    }
    return { ...result, faces };
  }, []);

  const analyzeStill = useCallback(async (el) => {
    if (!el) return;
    mediaRef.current = el;
    const r = await analyze(el, { detSize: 640 });
    setFrame(interpret(r, { live: false }));
    setSelected(0);
  }, [interpret]);

  // Re-interpret the current still when the gallery or calibration changes
  // (e.g. after naming a face), so labels and distances update immediately.
  useEffect(() => {
    if (mode !== "camera" && frame) {
      setFrame((fr) => (fr ? interpret({ ...fr, faces: fr.faces.map(({ embedding, bbox, kps, score, quality }) => ({ embedding, bbox, kps, score, quality })) }, { live: false }) : fr));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gallery, focal]);

  const stopCamera = useCallback(() => {
    loopRef.current++;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setFps(null);
  }, []);

  const startCamera = useCallback(async (videoEl) => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" }, audio: false });
      streamRef.current = stream;
      videoEl.srcObject = stream;
      await videoEl.play();
      mediaRef.current = videoEl;
      trackerRef.current.reset();
      setMode("camera");
      const token = ++loopRef.current;
      let last = performance.now();
      let smoothed = null;
      while (loopRef.current === token && streamRef.current) {
        if (videoEl.readyState >= 2) {
          // Detect every frame; re-identify a face only when its track is new,
          // every REID_EVERY frames, or while enrolling. In between, the
          // tracker carries the faceprint (and so the name) forward.
          const det = await detectFaces(videoEl, { detSize: 320, maxFaces: 6 });
          if (loopRef.current !== token) break;
          const tracker = trackerRef.current;
          const ids = tracker.update(det.faces.map((f) => f.bbox));
          const need = [];
          ids.forEach((id, i) => {
            const t = tracker.tracks.get(id);
            if (!t.embedding || t.age >= REID_EVERY || enrollRef.current) need.push(i);
          });
          const { embeddings, quality, embedMs } = need.length ? await embedFaces(need) : { embeddings: [], quality: [], embedMs: 0 };
          if (loopRef.current !== token) break;
          need.forEach((i, k) => Object.assign(tracker.tracks.get(ids[i]), { embedding: embeddings[k], quality: quality[k], age: 0 }));
          const faces = det.faces.map((f, i) => {
            const t = tracker.tracks.get(ids[i]);
            t.age = (t.age ?? 0) + 1;
            return { ...f, embedding: t.embedding, quality: t.quality };
          });
          const r = {
            faces, width: det.width, height: det.height,
            timings: { detectMs: det.detectMs, embedMs, reidentified: need.length, totalMs: +(det.detectMs + embedMs).toFixed(1) },
          };
          setFrame(interpret(r, { live: true, ids }));
          const now = performance.now();
          const inst = 1000 / (now - last);
          smoothed = smoothed == null ? inst : smoothed * 0.85 + inst * 0.15;
          last = now;
          setFps(Math.round(smoothed));
        } else {
          await new Promise((res) => setTimeout(res, 30));
        }
      }
    } catch (e) {
      setCameraError(e?.name === "NotAllowedError" ? "Camera permission was declined. You can still try the sample photo or upload one." : "No camera is available on this device. Try the sample photo or upload one.");
      stopCamera();
    }
  }, [interpret, stopCamera]);

  const useSample = useCallback(() => { stopCamera(); setPhotoUrl(null); setMode("sample"); }, [stopCamera]);
  const usePhoto = useCallback((file) => {
    stopCamera();
    setPhotoUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
    setMode("photo");
  }, [stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  /** Name a face. Stills enrol the visible face; live video collects several frames. */
  const enrollFace = useCallback((faceIndex, rawName) => {
    const name = rawName.trim().slice(0, 32);
    if (!name || !frame) return;
    if (mode === "camera") {
      const trackId = frame.faces[faceIndex]?.trackId;
      enrollRef.current = { name, needed: SAMPLES_PER_ENROLL, samples: [], frames: 0, trackId };
      setEnrolling({ name, got: 0, fix: null, trackId });
    } else {
      const f = frame.faces[faceIndex];
      if (!f) return;
      setGallery((prev) => ({ ...prev, [name]: [...(prev[name] || []), f.embedding] }));
      setTarget((t) => t || name);
      setNotice(f.check?.ok ? `Remembered ${name}.` : `Remembered ${name}, but the photo isn't ideal, so recognition may be weaker. Tip: ${f.check.checks.filter((c) => !c.ok).map((c) => c.fix.toLowerCase()).join("; ")}.`);
    }
  }, [frame, mode]);

  const cancelEnroll = useCallback(() => { enrollRef.current = null; setEnrolling(null); }, []);

  /** Calibrate the pinhole model: the visitor sits ~50 cm away; we solve for
   *  the focal length (in reference 960 px units) from their face width. */
  const calibrate = useCallback(() => {
    const faces = frame?.faces ?? [];
    if (!faces.length) { setNotice("Calibration needs your face in view."); return; }
    const f = faces.reduce((a, b) => (b.bbox[2] - b.bbox[0] > a.bbox[2] - a.bbox[0] ? b : a));
    const widthRef = (f.bbox[2] - f.bbox[0]) * (CONFIG.referenceWidthPx / frame.width);
    const value = Math.round((widthRef * CALIBRATION_CM) / CONFIG.knownFaceWidthCm);
    setFocal(value);
    setNotice(`Calibrated: your camera's focal length is about ${value} px (default 615). Distances now use it.`);
  }, [frame]);

  const exportData = useCallback(() => {
    const payload = {
      app: "VisionGuard", version: 1, model: "InsightFace buffalo_sc (w600k_mbf)", exportedAt: new Date().toISOString(),
      note: "Faceprints only: 512 numbers per sample. No images.",
      people: Object.fromEntries(Object.entries(gallery).map(([k, l]) => [k, l.map((e) => Array.from(e, (v) => +v.toFixed(5)))])),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "visionguard-faceprints.json" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [gallery]);

  const importData = useCallback(async (file) => {
    try {
      const data = JSON.parse(await file.text());
      const people = data.people || {};
      const clean = {};
      for (const [name, list] of Object.entries(people)) {
        const vecs = (Array.isArray(list) ? list : []).filter((v) => Array.isArray(v) && v.length === 512 && v.every(Number.isFinite));
        if (vecs.length) clean[String(name).slice(0, 32)] = vecs.map((v) => Float32Array.from(v));
      }
      if (!Object.keys(clean).length) throw new Error("no faceprints");
      setGallery((prev) => {
        const merged = { ...prev };
        for (const [k, l] of Object.entries(clean)) merged[k] = [...(merged[k] || []), ...l];
        return merged;
      });
      setNotice(`Imported ${Object.keys(clean).length} ${Object.keys(clean).length === 1 ? "person" : "people"}.`);
    } catch {
      setNotice("That file isn't a VisionGuard export.");
    }
  }, []);

  const forgetAll = useCallback(() => { setGallery({}); setTarget(""); setNotice("Everyone was forgotten. Nothing is stored now."); }, []);

  const forget = useCallback((name) => {
    setGallery((prev) => { const n = { ...prev }; delete n[name]; return n; });
    setTarget((t) => (t === name ? "" : t));
  }, []);

  const decision = useMemo(
    () => decide({ faces: frame?.faces ?? [], targetName: target, frameWidth: frame?.width ?? 960, emergencyStop: estop }),
    [frame, target, estop]
  );

  const value = {
    engine, mode, photoUrl, frame, gallery, target, estop, selected, enrolling, cameraError, fps, decision, mediaRef, focal, notice,
    setTarget, setEstop, setSelected, analyzeStill, startCamera, stopCamera, useSample, usePhoto, enrollFace, forget,
    cancelEnroll, calibrate, resetCalibration: () => setFocal(null), exportData, importData, forgetAll, setNotice,
  };
  return <VisionContext.Provider value={value}>{children}</VisionContext.Provider>;
}

export const useVision = () => useContext(VisionContext);
