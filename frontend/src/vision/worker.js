/**
 * Inference worker: runs both InsightFace ONNX models with onnxruntime-web
 * (WebAssembly) off the main thread, so the page stays at 60 fps while the
 * network runs. Frames arrive as transferred ImageBitmaps and never leave
 * this device.
 */
import * as ort from "onnxruntime-web/wasm";
import { ARCFACE_TEMPLATE, decodeScrfd, estimateSimilarity, l2normalize, letterbox, nms, rescale } from "./core.js";

let detector = null;
let recognizer = null;

async function init(base) {
  ort.env.wasm.wasmPaths = `${base}ort/`;
  ort.env.wasm.numThreads = 1; // no SharedArrayBuffer needed, works on any static host
  ort.env.logLevel = "error";
  // logSeverityLevel 3 = errors only (SCRFD's dynamic output shapes trigger harmless warnings)
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 };
  const t0 = performance.now();
  [detector, recognizer] = await Promise.all([
    ort.InferenceSession.create(`${base}models/det_500m.onnx`, opts),
    ort.InferenceSession.create(`${base}models/w600k_mbf.onnx`, opts),
  ]);
  return { loadMs: Math.round(performance.now() - t0) };
}

function detectInput(bitmap, size) {
  const { newW, newH, scale } = letterbox(bitmap.width, bitmap.height, size);
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, 0, 0, newW, newH);
  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    tensor[i] = (data[i * 4] - 127.5) / 128; // R
    tensor[plane + i] = (data[i * 4 + 1] - 127.5) / 128; // G
    tensor[2 * plane + i] = (data[i * 4 + 2] - 127.5) / 128; // B
  }
  return { tensor: new ort.Tensor("float32", tensor, [1, 3, size, size]), scale };
}

/** Warp each face onto the 112x112 ArcFace template and stack into one batch. */
function alignBatch(bitmap, faces) {
  const canvas = new OffscreenCanvas(112, 112);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  const plane = 112 * 112;
  const batch = new Float32Array(faces.length * 3 * plane);
  faces.forEach((face, n) => {
    const [a, b, tx, ty] = estimateSimilarity(face.kps, ARCFACE_TEMPLATE);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 112, 112);
    ctx.setTransform(a, b, -b, a, tx, ty);
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, 112, 112);
    const off = n * 3 * plane;
    for (let i = 0; i < plane; i++) {
      batch[off + i] = (data[i * 4] - 127.5) / 127.5;
      batch[off + plane + i] = (data[i * 4 + 1] - 127.5) / 127.5;
      batch[off + 2 * plane + i] = (data[i * 4 + 2] - 127.5) / 127.5;
    }
  });
  return new ort.Tensor("float32", batch, [faces.length, 3, 112, 112]);
}

let lastBitmap = null;
let lastFaces = [];

async function detect(bitmap, detSize, maxFaces) {
  const t0 = performance.now();
  const { tensor, scale } = detectInput(bitmap, detSize);
  const out = await detector.run({ [detector.inputNames[0]]: tensor });
  const outputs = detector.outputNames.map((n) => out[n].data);
  const faces = nms(decodeScrfd(outputs, detSize)).map((f) => rescale(f, scale)).slice(0, maxFaces);
  lastBitmap?.close?.();
  lastBitmap = bitmap; // kept so embed() can reuse this exact frame
  lastFaces = faces;
  return { faces, width: bitmap.width, height: bitmap.height, detectMs: +(performance.now() - t0).toFixed(1) };
}

/** Faceprints for chosen faces of the most recent detect() frame. */
async function embed(indices) {
  const t0 = performance.now();
  const chosen = indices.map((i) => lastFaces[i]).filter(Boolean);
  if (!chosen.length || !lastBitmap) return { embeddings: [], embedMs: 0 };
  const batch = alignBatch(lastBitmap, chosen);
  const emb = await recognizer.run({ [recognizer.inputNames[0]]: batch });
  const raw = emb[recognizer.outputNames[0]].data;
  const embeddings = chosen.map((_, n) => l2normalize(raw.subarray(n * 512, (n + 1) * 512)));
  return { embeddings, embedMs: +(performance.now() - t0).toFixed(1) };
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === "init") {
      self.postMessage({ id, ok: true, result: await init(e.data.base) });
    } else if (type === "detect") {
      self.postMessage({ id, ok: true, result: await detect(e.data.bitmap, e.data.detSize ?? 320, e.data.maxFaces ?? 10) });
    } else if (type === "embed") {
      const result = await embed(e.data.indices);
      self.postMessage({ id, ok: true, result }, result.embeddings.map((x) => x.buffer));
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
