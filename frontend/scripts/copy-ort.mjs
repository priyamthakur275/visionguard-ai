// Copies the onnxruntime-web WebAssembly runtime into public/ort so it is
// served from our own origin (no CDN), next to the ONNX models.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const dist = dirname(require.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"));
const out = new URL("../public/ort/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  cpSync(join(dist, f), join(out, f));
}
console.log("onnxruntime-web runtime copied to public/ort");
