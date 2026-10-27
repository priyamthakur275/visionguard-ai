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
