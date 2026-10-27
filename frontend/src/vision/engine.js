/**
 * Main-thread handle to the inference worker. One shared instance; calls
 * are promise-based and frames are transferred (zero-copy) to the worker.
 */
let worker = null;
let ready = null;
let seq = 0;
const pending = new Map();

function call(type, payload = {}, transfer = []) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

export function loadEngine() {
  if (ready) return ready;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    e.data.ok ? p.resolve(e.data.result) : p.reject(new Error(e.data.error));
  };
  ready = call("init", { base: new URL(import.meta.env.BASE_URL, location.href).href });
  return ready;
}

/** Step 1: find faces (boxes, scores, landmarks) in a frame. */
export async function detectFaces(source, { detSize = 320, maxFaces = 10 } = {}) {
  await loadEngine();
  const bitmap = await createImageBitmap(source);
  return call("detect", { bitmap, detSize, maxFaces }, [bitmap]);
}

/** Step 2: faceprints for selected faces of the frame passed to detectFaces. */
export function embedFaces(indices) {
  return call("embed", { indices });
}

/** Both steps for every face: used for photos. */
export async function analyze(source, opts) {
  const det = await detectFaces(source, opts);
  const { embeddings, embedMs } = await embedFaces(det.faces.map((_, i) => i));
  return {
    faces: det.faces.map((f, i) => ({ ...f, embedding: embeddings[i] })),
    width: det.width,
    height: det.height,
    timings: { detectMs: det.detectMs, embedMs, totalMs: +(det.detectMs + embedMs).toFixed(1) },
  };
}
