import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Camera, Check, ImageUp, Ruler, ScanFace, Square, X } from "lucide-react";
import { useVision } from "../state/VisionContext.jsx";
import { COLD, thermal } from "./thermal.js";
import Faceprint from "./Faceprint.jsx";

const SAMPLE = `${import.meta.env.BASE_URL}samples/astronaut.jpg`;

/** Where the media actually sits inside a box with object-fit: contain. */
function containRect(boxW, boxH, w, h) {
  const s = Math.min(boxW / w, boxH / h);
  return { s, ox: (boxW - w * s) / 2, oy: (boxH - h * s) / 2 };
}

function drawOverlay(canvas, frame, selected) {
  const box = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = box.width * dpr;
  canvas.height = box.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  if (!frame) return [];
  const { s, ox, oy } = containRect(box.width, box.height, frame.width, frame.height);
  const hit = [];
  frame.faces.forEach((f, i) => {
    const [x1, y1, x2, y2] = f.bbox.map((v, k) => (k % 2 === 0 ? ox + v * s : oy + v * s));
    const w = x2 - x1, h = y2 - y1;
    const color = f.isKnown ? thermal(0.55 + (f.similarity - 0.45) * 1.2) : COLD;
    const corner = Math.max(10, Math.min(w, h) * 0.22);
    ctx.lineWidth = i === selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x2, y2, -1, -1], [x1, y2, 1, -1]]) {
      ctx.moveTo(cx + dx * corner, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * corner);
    }
    ctx.stroke();
    if (i === selected) {
      ctx.fillStyle = f.isKnown ? thermal(0.6, 0.1) : "rgba(110,100,136,0.12)";
      ctx.fillRect(x1, y1, w, h);
    }
    ctx.fillStyle = color;
    for (const [kx, ky] of f.kps) {
      ctx.beginPath();
      ctx.arc(ox + kx * s, oy + ky * s, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    const label = f.isKnown ? `${f.name}  ${f.similarity.toFixed(2)}` : "Unknown";
    ctx.font = "600 13px 'Archivo Variable', system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 14;
    const ty = y1 - 26 < 2 ? y2 + 4 : y1 - 26;
    ctx.fillStyle = f.isKnown ? color : "rgba(23,18,28,0.85)";
    ctx.beginPath();
    ctx.roundRect(x1, ty, tw, 22, 6);
    ctx.fill();
    ctx.fillStyle = f.isKnown ? "#17121C" : "#E9E3EE";
    ctx.fillText(label, x1 + 7, ty + 15.5);
    if (f.distanceCm > 0) {
      ctx.font = "500 12px 'Archivo Variable', system-ui, sans-serif";
      ctx.fillStyle = "rgba(243,238,245,0.85)";
      ctx.fillText(`${Math.round(f.distanceCm)} cm`, x1, y2 + (ty > y2 ? 40 : 16));
    }
    hit.push({ i, x1, y1, x2, y2 });
  });
  return hit;
}

function Notice({ text, onDone }) {
  useEffect(() => {
    const t = setTimeout(onDone, 7000);
    return () => clearTimeout(t);
  }, [text, onDone]);
  return <p className="notice" role="status">{text}</p>;
}

export default function Viewfinder() {
  const v = useVision();
  const { frame, mode, selected, engine } = v;
  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const hitRef = useRef([]);
  const fileRef = useRef(null);
  const [name, setName] = useState("");

  const redraw = useCallback(() => {
    if (canvasRef.current) hitRef.current = drawOverlay(canvasRef.current, frame, selected);
  }, [frame, selected]);

  useLayoutEffect(redraw, [redraw]);
  useEffect(() => {
    const ro = new ResizeObserver(redraw);
    if (canvasRef.current) ro.observe(canvasRef.current.parentElement);
    return () => ro.disconnect();
  }, [redraw]);

  const src = mode === "photo" ? v.photoUrl : SAMPLE;
  const onImageLoad = () => engine.status === "ready" && v.analyzeStill(imgRef.current);
  useEffect(() => {
    if (engine.status === "ready" && mode !== "camera" && imgRef.current?.complete) v.analyzeStill(imgRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine.status, mode, src]);

  const onCanvasClick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const hit = hitRef.current.find((h) => x >= h.x1 && x <= h.x2 && y >= h.y1 && y <= h.y2);
    if (hit) v.setSelected(hit.i);
  };

  const enrolTarget = v.enrolling?.trackId != null ? frame?.faces?.find((f) => f.trackId === v.enrolling.trackId) : null;
  const face = enrolTarget ?? frame?.faces?.[selected] ?? frame?.faces?.[0];
  const submit = (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    v.enrollFace(frame.faces.indexOf(face), name);
    setName("");
  };

  return (
    <div className="viewfinder-wrap">
      <div className={`viewfinder ${mode === "camera" ? "is-live" : ""}`}>
        <img
          ref={imgRef}
          src={src}
          alt={mode === "photo" ? "Your uploaded photo" : "Sample photo: NASA astronaut portrait (public domain)"}
          onLoad={onImageLoad}
          hidden={mode === "camera"}
          crossOrigin="anonymous"
        />
        <video ref={videoRef} playsInline muted hidden={mode !== "camera"} />
        <canvas ref={canvasRef} className="overlay" onClick={onCanvasClick} />
        {engine.status === "loading" && <div className="vf-veil">Loading the face models (16 MB, once)…</div>}
        {engine.status === "error" && <div className="vf-veil">This browser couldn't start WebAssembly: {engine.error}</div>}
        <div className="vf-meta">
          <span className={`dot ${mode === "camera" ? "dot-live" : ""}`} />
          {mode === "camera" ? "Live camera" : mode === "photo" ? "Your photo" : "Sample photo"}
          {frame && <> · {frame.faces.length} face{frame.faces.length === 1 ? "" : "s"}</>}
        </div>
      </div>

      <div className="vf-controls">
        {mode === "camera" ? (
          <button className="btn btn-hot" onClick={v.useSample}><Square size={16} /> Stop camera</button>
        ) : (
          <button className="btn btn-hot" onClick={() => v.startCamera(videoRef.current)} disabled={engine.status !== "ready"}>
            <Camera size={16} /> Use my camera
          </button>
        )}
        <button className="btn" onClick={() => fileRef.current.click()} disabled={engine.status !== "ready"}>
          <ImageUp size={16} /> Upload a photo
        </button>
        {mode === "camera" && (
          <button className="btn btn-quiet" onClick={v.calibrate} title="Sit about 50 cm (an arm's length) from the camera, then click">
            <Ruler size={16} /> {v.focal ? "Recalibrate distance" : "Calibrate distance"}
          </button>
        )}
        {mode !== "sample" && <button className="btn btn-quiet" onClick={v.useSample}>Sample photo</button>}
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && v.usePhoto(e.target.files[0])} />
      </div>
      {v.cameraError && <p className="vf-error">{v.cameraError}</p>}
      {mode === "camera" && !v.focal && <p className="vf-hint">For accurate distance, sit about 50 cm away (an arm's length) and press Calibrate distance.</p>}
      {v.notice && <Notice text={v.notice} onDone={() => v.setNotice(null)} />}

      <div className="readout">
        <dl className="stats">
          <div><dt>Find faces</dt><dd>{frame ? `${frame.timings.detectMs} ms` : "–"}</dd></div>
          <div><dt>Faceprints</dt><dd>{!frame ? "–" : mode === "camera" && frame.timings.reidentified === 0 ? "reused" : `${frame.timings.embedMs} ms`}</dd></div>
          <div><dt>{mode === "camera" ? "Frame rate" : "Models loaded"}</dt><dd>{mode === "camera" ? (v.fps ? `${v.fps} fps` : "–") : engine.loadMs ? `${(engine.loadMs / 1000).toFixed(1)} s` : "–"}</dd></div>
        </dl>

        {face ? (
          <div className="face-card">
            <Faceprint embedding={face.embedding} cols={32} cell={6} />
            <div className="face-card-body">
              <p className="face-name">{face.isKnown ? face.name : "Unknown face"}</p>
              <p className="face-sub">
                {face.isKnown
                  ? `Match ${face.similarity.toFixed(2)} (needs 0.45) · ${Math.round(face.distanceCm)} cm`
                  : face.bestCandidate
                    ? `Closest: ${face.bestCandidate} at ${face.similarity.toFixed(2)}, below 0.45 · ${Math.round(face.distanceCm)} cm`
                    : `Nobody enrolled yet · ${Math.round(face.distanceCm)} cm`}
              </p>
              {face.check && (
                <ul className="checks" aria-label="Face quality">
                  {face.check.checks.map((c) => (
                    <li key={c.key} className={c.ok ? "ok" : "bad"} title={c.ok ? c.label : c.fix}>
                      {c.ok ? <Check size={13} strokeWidth={3} /> : <X size={13} strokeWidth={3} />} {c.label}
                    </li>
                  ))}
                </ul>
              )}
              {v.enrolling ? (
                <div className="enrol">
                  <div className="enrol-top">
                    <span>Learning <b>{v.enrolling.name}</b>: {v.enrolling.got} of 5 good frames</span>
                    <button className="link-btn" onClick={v.cancelEnroll}>Cancel</button>
                  </div>
                  <div className="enrol-bar"><span style={{ width: `${(v.enrolling.got / 5) * 100}%` }} /></div>
                  <p className="face-sub hot">{v.enrolling.fix || "Good. Keep still, then turn your head a little."}</p>
                </div>
              ) : (
                <form className="name-form" onSubmit={submit}>
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={face.isKnown ? `Add more of ${face.name}` : "Name this face"}
                    aria-label="Name for this face"
                    maxLength={32}
                  />
                  <button className="btn btn-small" disabled={!name.trim()}><ScanFace size={15} /> Remember</button>
                </form>
              )}
            </div>
          </div>
        ) : (
          frame && <p className="face-sub">No face found. Try a well-lit, front-facing photo.</p>
        )}
      </div>
    </div>
  );
}
