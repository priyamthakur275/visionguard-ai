import { useEffect, useMemo, useRef } from "react";
import { useVision } from "../state/VisionContext.jsx";
import { pca2 } from "../vision/core.js";
import { thermal } from "./thermal.js";

const W = 640;
const H = 400;
const PAD = 48;

/** Colour for person i of n, spread along the thermal ramp (skipping the darkest end). */
const personColour = (i, n) => thermal(0.35 + (0.65 * (i + 0.5)) / Math.max(n, 1));

export default function FaceSpace() {
  const { gallery, frame, selected } = useVision();
  const canvasRef = useRef(null);
  const names = Object.keys(gallery);
  const samples = useMemo(() => names.flatMap((n) => gallery[n].map((v) => ({ name: n, v }))), [gallery]);
  const face = frame?.faces?.[selected] ?? frame?.faces?.[0];
  const live = face?.embedding?.length === 512 ? face.embedding : null;

  const space = useMemo(() => {
    if (samples.length < 2) return null;
    const pca = pca2(samples.map((s) => s.v));
    const pts = samples.map((s) => ({ ...s, p: pca.project(s.v) }));
    const xs = pts.map((p) => p.p[0]);
    const ys = pts.map((p) => p.p[1]);
    const span = (a) => Math.max(Math.max(...a) - Math.min(...a), 1e-3);
    // Equal scale on both axes, so distances on screen mean the same thing in every direction.
    const s = Math.min((W - 2 * PAD) / span(xs), (H - 2 * PAD) / span(ys)) * 0.8;
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const toScreen = ([x, y]) => [W / 2 + (x - cx) * s, H / 2 - (y - cy) * s];
    const shown = (pca.variances[0] + pca.variances[1]) / (pca.totalVariance || 1);
    return { pca, pts, toScreen, shown };
  }, [samples]);

  const livePt = space && live ? space.toScreen(space.pca.project(live)) : null;
  const clamp = (p) => p && [Math.min(W - 12, Math.max(12, p[0])), Math.min(H - 12, Math.max(12, p[1]))];
  const liveXY = clamp(livePt);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    c.width = W * dpr;
    c.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // faint grid
    ctx.strokeStyle = "rgba(164,150,174,0.10)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 40) { ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke(); }
    if (!space) return;

    const groups = names.map((n, i) => ({ n, col: personColour(i, names.length), pts: space.pts.filter((p) => p.name === n).map((p) => space.toScreen(p.p)) }));
    for (const g of groups) {
      for (const [x, y] of g.pts) {
        ctx.fillStyle = g.col;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    // line from the live face to the person it matched
    if (liveXY && face?.isKnown) {
      const g = groups.find((gg) => gg.n === face.name);
      if (g?.pts.length) {
        const [mx, my] = g.pts.reduce(([sx, sy], [x, y]) => [sx + x / g.pts.length, sy + y / g.pts.length], [0, 0]);
        ctx.setLineDash([4, 5]);
        ctx.strokeStyle = "rgba(243,238,245,0.55)";
        ctx.beginPath(); ctx.moveTo(liveXY[0], liveXY[1]); ctx.lineTo(mx, my); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.font = "600 13px 'Archivo Variable', system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const g of groups) {
      if (!g.pts.length) continue;
      const [mx, my] = g.pts.reduce(([sx, sy], [x, y]) => [sx + x / g.pts.length, sy + y / g.pts.length], [0, 0]);
      const label = g.n;
      const tw = ctx.measureText(label).width;
      const lx = Math.min(W - tw - 18, Math.max(6, mx + 12));
      const ly = Math.min(H - 14, Math.max(14, my - 16));
      ctx.fillStyle = "rgba(23,18,28,0.85)";
      ctx.fillRect(lx - 6, ly - 11, tw + 12, 22);
      ctx.fillStyle = g.col;
      ctx.fillText(label, lx, ly);
    }
  }, [space, liveXY?.[0], liveXY?.[1], face?.name, face?.isKnown]);

  return (
    <section id="space" className="section">
      <div className="section-head">
        <h2>What the computer sees: a map of faces</h2>
        <p>
          Each dot is one saved faceprint. Photos of the same person land close together; different people sit apart.
          Your live face is the white ring, and recognition simply asks which group it's nearest to.
        </p>
      </div>
      <div className="space-wrap">
        <div className="space-canvas">
          <canvas ref={canvasRef} style={{ width: "100%", aspectRatio: `${W} / ${H}` }} aria-label="Two-dimensional map of saved faceprints" />
          {liveXY && (
            <span className="space-live" style={{ left: `${(liveXY[0] / W) * 100}%`, top: `${(liveXY[1] / H) * 100}%` }} aria-hidden="true" />
          )}
          {!space && (
            <p className="space-empty">
              {samples.length === 0 ? "Name a face above and its faceprints appear here." : "One more faceprint and the map can be drawn. Add another person, or more of the same one."}
            </p>
          )}
        </div>
        <aside className="space-notes">
          <p>
            <b>How the map is made.</b> A faceprint has 512 numbers, which can't be drawn. Principal component analysis
            finds the two directions in which your saved faceprints differ most and flattens everything onto them.
          </p>
          {space && (
            <p>
              These two directions hold <b>{Math.round(Math.min(1, space.shown) * 100)}%</b> of the differences between your
              saved faceprints. The rest is lost in the flattening, so the actual match uses all 512 numbers, not this picture.
            </p>
          )}
          {face && (
            <p>
              Live face: {face.isKnown ? <><b>{face.name}</b>, similarity {face.similarity.toFixed(2)}</> : <>Unknown{face.bestCandidate ? <>, closest is {face.bestCandidate} at {face.similarity.toFixed(2)}</> : null}</>}.
            </p>
          )}
          {names.length === 1 && space && <p>Add a second person to see the groups pull apart.</p>}
          {names.length > 0 && (
            <ul className="space-legend">
              {names.map((n, i) => (
                <li key={n}><span style={{ background: personColour(i, names.length) }} />{n} <em>{gallery[n].length}</em></li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </section>
  );
}
