import { useEffect, useRef } from "react";
import { useVision } from "../state/VisionContext.jsx";
import { ARCFACE_TEMPLATE, estimateSimilarity } from "../vision/core.js";
import { CONFIG } from "../vision/follow.js";
import Faceprint from "./Faceprint.jsx";
import { COLD, thermal } from "./thermal.js";

/** The exact 112x112 crop the recogniser sees, rebuilt from the landmarks. */
function AlignedCrop({ face, media }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0E0B12";
    ctx.fillRect(0, 0, 112, 112);
    if (!face || !media) return;
    const [a, b, tx, ty] = estimateSimilarity(face.kps, ARCFACE_TEMPLATE);
    ctx.setTransform(a, b, -b, a, tx, ty);
    try {
      ctx.drawImage(media, 0, 0);
    } catch {
      /* media not ready yet */
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#FFD15C";
    for (const [x, y] of ARCFACE_TEMPLATE) {
      ctx.beginPath();
      ctx.arc(x, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  return <canvas ref={ref} width={112} height={112} className="crop" aria-label="Aligned 112 by 112 face crop" />;
}

export default function HowItWorks() {
  const { frame, selected, gallery, mediaRef, decision } = useVision();
  const face = frame?.faces?.[selected] ?? frame?.faces?.[0];
  const people = Object.keys(gallery);
  const w = face ? Math.round(face.bbox[2] - face.bbox[0]) : null;
  const focal = frame ? Math.round(CONFIG.focalLengthPx * (frame.width / CONFIG.referenceWidthPx)) : null;

  return (
    <section id="how" className="section">
      <div className="section-head">
        <h2>Five steps, all inside this tab</h2>
        <p>
          Every number below comes from the face currently selected above. Nothing is uploaded: the two neural
          networks run on your device through WebAssembly.
        </p>
      </div>

      <ol className="steps">
        <li className="step">
          <div className="step-num">1</div>
          <div className="step-body">
            <h3>Find the faces</h3>
            <p>
              A small detector network (SCRFD, 2.5 MB) scans the image and returns a box, a confidence score and five
              landmarks (eyes, nose, mouth corners) for every face.
            </p>
            <p className="live">
              {frame ? <>Found <b>{frame.faces.length}</b> face{frame.faces.length === 1 ? "" : "s"} in <b>{frame.timings.detectMs} ms</b>{face && <> · confidence <b>{face.score.toFixed(2)}</b></>}</> : "Waiting for an image…"}
            </p>
          </div>
        </li>

        <li className="step">
          <div className="step-num">2</div>
          <div className="step-body">
            <h3>Line it up</h3>
            <p>
              The five landmarks are rotated, scaled and shifted onto fixed reference points, so every face reaches the
              next network upright and the same size: 112 × 112 pixels.
            </p>
          </div>
          <AlignedCrop face={face} media={mediaRef.current} />
        </li>

        <li className="step">
          <div className="step-num">3</div>
          <div className="step-body">
            <h3>Turn it into 512 numbers</h3>
            <p>
              The recogniser (ArcFace, MobileFaceNet, 13 MB) outputs a <em>faceprint</em>: 512 numbers, scaled to length 1.
              Photos of the same person land close together; different people land far apart. Only these numbers are
              ever stored, never the photo.
            </p>
          </div>
          <Faceprint embedding={face?.embedding} cols={32} cell={7} />
        </li>

        <li className="step">
          <div className="step-num">4</div>
          <div className="step-body">
            <h3>Compare with people you named</h3>
            <p>
              Similarity is the dot product of two faceprints (cosine similarity). At <b>0.45</b> or above it's a match;
              below, the face stays Unknown. Unrelated faces typically score under 0.25.
            </p>
            {face && people.length > 0 ? (
              <div className="bars">
                {people.map((p) => {
                  const s = face.scores?.[p] ?? 0;
                  const pass = s >= CONFIG.similarityThreshold;
                  return (
                    <div className="bar-row" key={p}>
                      <span className="bar-name">{p}</span>
                      <span className="bar-track">
                        <span className="bar-fill" style={{ width: `${Math.max(0, s) * 100}%`, background: pass ? thermal(0.55 + (s - 0.45)) : COLD }} />
                        <span className="bar-threshold" style={{ left: "45%" }} />
                      </span>
                      <span className="bar-val">{s.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="live">Name a face above and its similarity bars appear here.</p>
            )}
          </div>
        </li>

        <li className="step">
          <div className="step-num">5</div>
          <div className="step-body">
            <h3>Measure distance, then decide</h3>
            <p>
              Faces are about 14 cm wide, so the pinhole camera model gives distance from the box width:
              distance = 14 cm × focal length ÷ face width in pixels.
            </p>
            <p className="live">
              {face ? (
                <>14 × {focal} ÷ {w} px = <b>{Math.round(face.distanceCm)} cm</b> · robot: <b>{decision.command}</b></>
              ) : "Waiting for a face…"}
            </p>
          </div>
        </li>
      </ol>
    </section>
  );
}
