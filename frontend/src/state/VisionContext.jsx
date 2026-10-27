import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { analyze, detectFaces, embedFaces, loadEngine } from "../vision/engine.js";
import { matchIdentity } from "../vision/core.js";
import { CONFIG, CentroidTracker, decide, distanceZone, estimateDistanceCm } from "../vision/follow.js";

const VisionContext = createContext(null);
const STORE_KEY = "visionguard.gallery.v1";
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
      return { ...f, ...m, name: label, isKnown, trackId: ids[i], distanceCm, zone: distanceZone(distanceCm) };
    });

    // Multi-frame enrolment: collect samples from the largest face.
    const job = enrollRef.current;
    if (job && faces.length) {
      const biggest = faces.reduce((a, b) => (b.bbox[2] - b.bbox[0] > a.bbox[2] - a.bbox[0] ? b : a));
      job.samples.push(biggest.embedding);
      setEnrolling({ name: job.name, got: job.samples.length });
      if (job.samples.length >= job.needed) {
        enrollRef.current = null;
        setEnrolling(null);
        setGallery((prev) => ({ ...prev, [job.name]: [...(prev[job.name] || []), ...job.samples] }));
        setTarget((t) => t || job.name);
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

  // Re-label the current still when the gallery changes (e.g. after naming a face).
  useEffect(() => {
    if (mode !== "camera" && frame) {
      setFrame((fr) => (fr ? interpret({ ...fr, faces: fr.faces.map(({ embedding, bbox, kps, score }) => ({ embedding, bbox, kps, score })) }, { live: false }) : fr));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gallery]);

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
          const { embeddings, embedMs } = need.length ? await embedFaces(need) : { embeddings: [], embedMs: 0 };
          if (loopRef.current !== token) break;
          need.forEach((i, k) => Object.assign(tracker.tracks.get(ids[i]), { embedding: embeddings[k], age: 0 }));
          const faces = det.faces.map((f, i) => {
            const t = tracker.tracks.get(ids[i]);
            t.age = (t.age ?? 0) + 1;
            return { ...f, embedding: t.embedding };
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
      enrollRef.current = { name, needed: SAMPLES_PER_ENROLL, samples: [] };
      setEnrolling({ name, got: 0 });
    } else {
      const f = frame.faces[faceIndex];
      if (!f) return;
      setGallery((prev) => ({ ...prev, [name]: [...(prev[name] || []), f.embedding] }));
      setTarget((t) => t || name);
    }
  }, [frame, mode]);

  const forget = useCallback((name) => {
    setGallery((prev) => { const n = { ...prev }; delete n[name]; return n; });
    setTarget((t) => (t === name ? "" : t));
  }, []);

  const decision = useMemo(
    () => decide({ faces: frame?.faces ?? [], targetName: target, frameWidth: frame?.width ?? 960, emergencyStop: estop }),
    [frame, target, estop]
  );

  const value = {
    engine, mode, photoUrl, frame, gallery, target, estop, selected, enrolling, cameraError, fps, decision, mediaRef,
    setTarget, setEstop, setSelected, analyzeStill, startCamera, stopCamera, useSample, usePhoto, enrollFace, forget,
  };
  return <VisionContext.Provider value={value}>{children}</VisionContext.Provider>;
}

export const useVision = () => useContext(VisionContext);
