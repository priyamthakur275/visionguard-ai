import { useEffect, useRef } from "react";
import { thermal } from "./thermal.js";

/**
 * Draws a 512-number ArcFace embedding as a 32 x 16 heat grid. Each cell is
 * one number; similar faces produce visibly similar patterns.
 */
export default function Faceprint({ embedding, cols = 32, cell = 9, label }) {
  const ref = useRef(null);
  const rows = Math.ceil(512 / cols);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = cols * cell * dpr;
    c.height = rows * cell * dpr;
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cols * cell, rows * cell);
    for (let i = 0; i < 512; i++) {
      const v = embedding ? embedding[i] : 0;
      // Embedding values sit roughly in [-0.15, 0.15].
      ctx.fillStyle = embedding ? thermal(0.5 + v / 0.3) : "rgba(255,255,255,0.05)";
      const x = (i % cols) * cell;
      const y = Math.floor(i / cols) * cell;
      ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
    }
  }, [embedding, cols, cell, rows]);

  return (
    <canvas
      ref={ref}
      className="faceprint"
      style={{ width: cols * cell, height: rows * cell }}
      role="img"
      aria-label={label || (embedding ? "Faceprint: 512 numbers describing this face" : "No face selected")}
    />
  );
}
