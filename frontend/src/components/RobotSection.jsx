import { useEffect, useRef } from "react";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, OctagonX, Pause } from "lucide-react";
import { useVision } from "../state/VisionContext.jsx";
import { CONFIG } from "../vision/follow.js";
import { COLD, thermal } from "./thermal.js";

const FOV_DEG = 60; // typical laptop webcam horizontal field of view
const MAX_CM = 200;

const ICONS = { LEFT: ArrowLeft, RIGHT: ArrowRight, FORWARD: ArrowUp, BACKWARD: ArrowDown, STOP: Pause };

const FAILSAFES = [
  { key: "unknown-lockout", title: "Unknown faces can't drive", body: "Only an enrolled, matched face can ever produce a movement command." },
  { key: "target-lost", title: "Target lost means stop", body: "The moment the chosen person leaves the frame, the robot holds still." },
  { key: "emergency-stop", title: "Emergency stop wins", body: "The stop button overrides every other rule, every frame." },
  { key: "no-target", title: "No target, no motion", body: "Nothing moves until a person is explicitly chosen to follow." },
];

function drawField(canvas, frame, target, decision) {
  const box = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = box.width, H = box.height;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const rx = W / 2, ry = H - 34;
  const pxPerCm = (H - 70) / MAX_CM;
  const half = (FOV_DEG / 2) * (Math.PI / 180);
  const toXY = (bearingRad, cm) => [rx + Math.sin(bearingRad) * cm * pxPerCm, ry - Math.cos(bearingRad) * cm * pxPerCm];

  // Field of view
  ctx.fillStyle = "rgba(255,255,255,0.025)";
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.arc(rx, ry, MAX_CM * pxPerCm, -Math.PI / 2 - half, -Math.PI / 2 + half);
  ctx.closePath();
  ctx.fill();

  // Ideal following band (50-100 cm)
  ctx.fillStyle = "rgba(242,84,45,0.09)";
  ctx.beginPath();
  ctx.arc(rx, ry, CONFIG.idealMaxCm * pxPerCm, -Math.PI / 2 - half, -Math.PI / 2 + half);
  ctx.arc(rx, ry, CONFIG.backwardCm * pxPerCm, -Math.PI / 2 + half, -Math.PI / 2 - half, true);
  ctx.closePath();
  ctx.fill();

  // Range rings + labels
  ctx.strokeStyle = "rgba(255,255,255,0.09)";
  ctx.fillStyle = "rgba(243,238,245,0.45)";
  ctx.font = "500 11px 'Archivo Variable', system-ui";
  for (const cm of [50, 100, 150, 200]) {
    ctx.beginPath();
    ctx.arc(rx, ry, cm * pxPerCm, -Math.PI / 2 - half, -Math.PI / 2 + half);
    ctx.stroke();
    const [lx, ly] = toXY(half, cm);
    ctx.fillText(`${cm} cm`, lx + 6, ly + 4);
  }

  // Centre dead zone (±60 px of a 960 px frame)
  const dz = ((CONFIG.deadZonePx / CONFIG.referenceWidthPx) * FOV_DEG * Math.PI) / 180;
  ctx.fillStyle = "rgba(255,209,92,0.07)";
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.arc(rx, ry, MAX_CM * pxPerCm, -Math.PI / 2 - dz, -Math.PI / 2 + dz);
  ctx.closePath();
  ctx.fill();

  // People
  const faces = frame?.faces ?? [];
  faces.forEach((f) => {
    if (!(f.distanceCm > 0)) return;
    const bearing = (((f.bbox[0] + f.bbox[2]) / 2 / frame.width) - 0.5) * FOV_DEG * (Math.PI / 180);
    const [x, y] = toXY(bearing, Math.min(f.distanceCm, MAX_CM - 4));
    const isTarget = f.isKnown && f.name === target;
    if (isTarget) {
      ctx.strokeStyle = thermal(0.75, 0.7);
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = isTarget ? thermal(0.8) : f.isKnown ? thermal(0.45) : COLD;
    ctx.beginPath();
    ctx.arc(x, y, isTarget ? 9 : 6, 0, Math.PI * 2);
    ctx.fill();
    if (f.isKnown) { // unknown people get a dot only, so crowds stay readable
      ctx.fillStyle = "rgba(243,238,245,0.9)";
      ctx.font = "600 12px 'Archivo Variable', system-ui";
      ctx.fillText(f.name, x + 12, y + 4);
    }
  });

  // Legend
  ctx.font = "500 11px 'Archivo Variable', system-ui";
  [["Following", thermal(0.8)], ["Known", thermal(0.45)], ["Unknown", COLD]].forEach(([label, color], i) => {
    const lx = 14, ly = H - 62 + i * 18;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx + 4, ly - 4, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(243,238,245,0.6)";
    ctx.fillText(label, lx + 14, ly);
  });

  // Robot
  ctx.save();
  ctx.translate(rx, ry);
  const turn = decision.command === "LEFT" ? -0.18 : decision.command === "RIGHT" ? 0.18 : 0;
  ctx.rotate(turn);
  ctx.fillStyle = "#2A2131";
  ctx.strokeStyle = decision.failsafe === "emergency-stop" ? "#FF5468" : thermal(0.7);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-18, -16, 36, 32, 9);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.beginPath();
  ctx.moveTo(0, -24);
  ctx.lineTo(6, -16);
  ctx.lineTo(-6, -16);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export default function RobotSection() {
  const v = useVision();
  const { frame, target, decision, gallery, estop } = v;
  const canvasRef = useRef(null);
  const names = Object.keys(gallery);
  const Icon = ICONS[decision.command];

  useEffect(() => {
    const c = canvasRef.current;
    const draw = () => c && drawField(c, frame, target, decision);
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c.parentElement);
    return () => ro.disconnect();
  }, [frame, target, decision]);

  return (
    <section id="robot" className="section">
      <div className="section-head">
        <h2>Follow one person. Ignore everyone else.</h2>
        <p>
          The same faces drive a robot. Pick who it should follow; it turns until they're centred, then keeps them
          50–100 cm away. These are the exact rules the desktop app sends to the Arduino.
        </p>
      </div>

      <div className="robot-grid">
        <div className="robot-field">
          <canvas ref={canvasRef} />
          <div className={`command ${decision.command.toLowerCase()} ${decision.failsafe === "emergency-stop" ? "is-estop" : ""}`}>
            <Icon size={22} strokeWidth={2.5} />
            <div>
              <p className="command-word">{decision.command}</p>
              <p className="command-why">{decision.reason}</p>
            </div>
          </div>
        </div>

        <div className="robot-side">
          <label className="field-label">
            Follow
            <select value={target} onChange={(e) => v.setTarget(e.target.value)} disabled={!names.length}>
              <option value="">{names.length ? "Choose a person" : "Name a face first"}</option>
              {names.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>

          <button
            className={`estop ${estop ? "on" : ""}`}
            onClick={() => v.setEstop(!estop)}
            aria-pressed={estop}
          >
            <OctagonX size={22} />
            {estop ? "Release emergency stop" : "Emergency stop"}
          </button>

          <ul className="failsafes">
            {FAILSAFES.map((f) => (
              <li key={f.key} className={decision.failsafe === f.key ? "active" : ""}>
                <span className="lamp" aria-hidden />
                <div>
                  <p className="fs-title">{f.title}{decision.failsafe === f.key && <span className="fs-now">active now</span>}</p>
                  <p className="fs-body">{f.body}</p>
                </div>
              </li>
            ))}
          </ul>
          {!frame?.faces?.length && <p className="hint">Start the camera above (or use a photo) to see the robot react.</p>}
        </div>
      </div>
    </section>
  );
}
