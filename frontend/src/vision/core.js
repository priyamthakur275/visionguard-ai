/**
 * Pure maths for the in-browser pipeline. No DOM, no ONNX: every function
 * here is deterministic and unit-tested in Node (see core.test.mjs).
 *
 * It is a line-by-line port of what InsightFace does in Python:
 *   - SCRFD output decoding + non-maximum suppression  (insightface/model_zoo/scrfd.py)
 *   - 5-point similarity alignment to the ArcFace 112x112 template (utils/face_align.py)
 *   - cosine matching against an enrolled gallery       (app/core/recognition_utils.py)
 */

export const DET_THRESHOLD = 0.5;
export const NMS_IOU = 0.4;
export const STRIDES = [8, 16, 32];
export const ANCHORS_PER_CELL = 2;

/** ArcFace reference landmarks for a 112x112 crop: eyes, nose, mouth corners. */
export const ARCFACE_TEMPLATE = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/** Letterbox geometry: scale the image to fit `size` x `size`, anchored top-left. */
export function letterbox(width, height, size) {
  const imRatio = height / width;
  let newW, newH;
  if (imRatio > 1) {
    newH = size;
    newW = Math.floor(newH / imRatio);
  } else {
    newW = size;
    newH = Math.floor(newW * imRatio);
  }
  return { newW, newH, scale: newH / height };
}

/**
 * Decode SCRFD's 9 output tensors (scores, box distances, keypoint offsets
 * for strides 8/16/32) into faces in letterboxed-input coordinates.
 * `outputs` is [s8, s16, s32, b8, b16, b32, k8, k16, k32] as Float32Arrays.
 */
export function decodeScrfd(outputs, inputSize, threshold = DET_THRESHOLD) {
  const faces = [];
  STRIDES.forEach((stride, i) => {
    const scores = outputs[i];
    const boxes = outputs[i + 3];
    const kps = outputs[i + 6];
    const cells = Math.ceil(inputSize / stride);
    let anchor = 0;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        for (let a = 0; a < ANCHORS_PER_CELL; a++, anchor++) {
          const score = scores[anchor];
          if (score < threshold) continue;
          const cx = x * stride;
          const cy = y * stride;
          const b = anchor * 4;
          const k = anchor * 10;
          const landmarks = [];
          for (let p = 0; p < 5; p++) {
            landmarks.push([cx + kps[k + 2 * p] * stride, cy + kps[k + 2 * p + 1] * stride]);
          }
          faces.push({
            score,
            bbox: [
              cx - boxes[b] * stride,
              cy - boxes[b + 1] * stride,
              cx + boxes[b + 2] * stride,
              cy + boxes[b + 3] * stride,
            ],
            kps: landmarks,
          });
        }
      }
    }
  });
  return faces;
}

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
  const areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  const areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return inter / (areaA + areaB - inter);
}

/** Greedy NMS, highest score first (same "+1" area convention as InsightFace). */
export function nms(faces, threshold = NMS_IOU) {
  const sorted = [...faces].sort((p, q) => q.score - p.score);
  const kept = [];
  for (const f of sorted) {
    if (kept.every((k) => iou(k.bbox, f.bbox) <= threshold)) kept.push(f);
  }
  return kept;
}

export function rescale(face, scale) {
  return {
    ...face,
    bbox: face.bbox.map((v) => v / scale),
    kps: face.kps.map(([x, y]) => [x / scale, y / scale]),
  };
}

/**
 * Least-squares similarity transform (rotation + uniform scale + shift)
 * mapping `src` points onto `dst`. Returns [a, b, tx, ty] for
 *   u = a·x − b·y + tx,   v = b·x + a·y + ty.
 * Same solution as skimage's SimilarityTransform.estimate (Umeyama), which
 * InsightFace uses, for the non-reflective case.
 */
export function estimateSimilarity(src, dst) {
  const n = src.length;
  let mx = 0, my = 0, mu = 0, mv = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0]; my += src[i][1]; mu += dst[i][0]; mv += dst[i][1];
  }
  mx /= n; my /= n; mu /= n; mv /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - mx, y = src[i][1] - my;
    const u = dst[i][0] - mu, v = dst[i][1] - mv;
    num1 += x * u + y * v;
    num2 += x * v - y * u;
    den += x * x + y * y;
  }
  const a = num1 / den;
  const b = num2 / den;
  return [a, b, mu - (a * mx - b * my), mv - (b * mx + a * my)];
}

export function l2normalize(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const inv = 1 / Math.sqrt(s || 1);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Match one normalised embedding against a gallery
 * { name: [Float32Array, ...] }. The best score over each person's samples
 * wins; below `threshold` the face is Unknown.
 */
export function matchIdentity(embedding, gallery, threshold) {
  let bestName = null;
  let best = -1;
  const scores = {};
  for (const [name, samples] of Object.entries(gallery)) {
    let personBest = -1;
    for (const s of samples) personBest = Math.max(personBest, dot(embedding, s));
    scores[name] = personBest;
    if (personBest > best) {
      best = personBest;
      bestName = name;
    }
  }
  const known = bestName !== null && best >= threshold;
  return { name: known ? bestName : "Unknown", isKnown: known, similarity: best, bestCandidate: bestName, scores };
}

// ---------------------------------------------------------------------------
// Face quality (used to accept or reject enrolment samples)
// ---------------------------------------------------------------------------

/** Brightness (mean grey) and sharpness (variance of a 4-neighbour Laplacian)
 *  of an aligned 112x112 RGBA crop. Calibrated on InsightFace crops: sharp
 *  faces score ~120-420; a slight blur (sigma 1.5) drops them to ~15-40. */
export function cropStats(rgba, size = 112) {
  const g = new Float32Array(size * size);
  let sum = 0;
  for (let i = 0; i < size * size; i++) {
    const v = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    g[i] = v;
    sum += v;
  }
  let n = 0, mean = 0, m2 = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const lap = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - size] - g[i + size];
      n++;
      const d = lap - mean;
      mean += d / n;
      m2 += d * (lap - mean);
    }
  }
  return { brightness: sum / (size * size), sharpness: m2 / n };
}

/** Head turn from landmarks: nose offset from the eye midpoint, in units of
 *  eye distance. ~0 = facing the camera; ±0.3 and beyond = clearly turned. */
export function yawRatio(kps) {
  const [le, re, nose] = kps;
  const eyeDist = Math.hypot(re[0] - le[0], re[1] - le[1]) || 1;
  return (nose[0] - (le[0] + re[0]) / 2) / eyeDist;
}

/** Head tilt in degrees from the line between the eyes. */
export function rollDeg(kps) {
  const [le, re] = kps;
  return (Math.atan2(re[1] - le[1], re[0] - le[0]) * 180) / Math.PI;
}

export const QUALITY = { minFacePx: 80, minSharpness: 60, minBrightness: 55, maxBrightness: 210, maxYaw: 0.45, maxRoll: 20 };

/** Checklist used before a face is remembered. Each item has a fix hint. */
export function faceQuality(face, frameWidth) {
  const widthPx = (face.bbox[2] - face.bbox[0]) * (960 / (frameWidth || 960));
  const yaw = yawRatio(face.kps);
  const roll = rollDeg(face.kps);
  const q = face.quality || {};
  const checks = [
    { key: "size", label: "Close enough", ok: widthPx >= QUALITY.minFacePx, fix: "Move closer to the camera" },
    { key: "sharp", label: "Sharp", ok: q.sharpness == null || q.sharpness >= QUALITY.minSharpness, fix: "Hold still: the face is blurry" },
    { key: "light", label: "Well lit", ok: q.brightness == null || (q.brightness >= QUALITY.minBrightness && q.brightness <= QUALITY.maxBrightness), fix: q.brightness > QUALITY.maxBrightness ? "Too bright: move away from the light" : "Too dark: face a light or window" },
    { key: "pose", label: "Facing camera", ok: Math.abs(yaw) <= QUALITY.maxYaw && Math.abs(roll) <= QUALITY.maxRoll, fix: Math.abs(roll) > QUALITY.maxRoll ? "Straighten your head" : "Turn a little towards the camera" },
  ];
  return { checks, ok: checks.every((c) => c.ok), yaw, roll, widthPx };
}

// ---------------------------------------------------------------------------
// Face space: 2-D principal component view of faceprints
// ---------------------------------------------------------------------------

/**
 * Top-2 principal components of `vectors` (n x d) via the n x n Gram matrix
 * and power iteration (n is small: the enrolled samples). Returns the mean,
 * two unit component vectors in d-space, and a project() for new vectors.
 */
export function pca2(vectors, iterations = 200) {
  const n = vectors.length;
  const d = vectors[0].length;
  const mean = new Float32Array(d);
  for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  const X = vectors.map((v) => Float32Array.from(v, (x, j) => x - mean[j]));
  const G = X.map((a) => X.map((b) => dot(a, b)));
  const totalVariance = G.reduce((t, row, i) => t + row[i], 0) / n;
  const comps = [];
  for (let k = 0; k < 2; k++) {
    let v = Array.from({ length: n }, (_, i) => 1 + ((i * 7919) % 13) / 13);
    let lambda = 0;
    for (let it = 0; it < iterations; it++) {
      const w = G.map((row) => row.reduce((s, g, j) => s + g * v[j], 0));
      lambda = Math.hypot(...w);
      if (lambda < 1e-9) break;
      v = w.map((x) => x / lambda);
    }
    // Back to d-space: pc = X^T v, normalised.
    const pc = new Float32Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) pc[j] += X[i][j] * v[i];
    const norm = Math.hypot(...pc) || 1;
    for (let j = 0; j < d; j++) pc[j] /= norm;
    comps.push({ pc, lambda });
    // Deflate the Gram matrix.
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i][j] -= lambda * v[i] * v[j];
  }
  const project = (vec) => comps.map(({ pc }) => {
    let s = 0;
    for (let j = 0; j < d; j++) s += (vec[j] - mean[j]) * pc[j];
    return s;
  });
  return { mean, components: comps.map((c) => c.pc), variances: comps.map((c) => c.lambda / n), totalVariance, project };
}
