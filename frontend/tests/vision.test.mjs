// Unit tests for the browser pipeline maths. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ARCFACE_TEMPLATE, decodeScrfd, estimateSimilarity, iou, l2normalize, letterbox, matchIdentity, nms,
} from "../src/vision/core.js";
import { CentroidTracker, LabelSmoother, decide, distanceZone, estimateDistanceCm, selectTarget } from "../src/vision/follow.js";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test("letterbox keeps aspect ratio", () => {
  assert.deepEqual(letterbox(1280, 720, 640), { newW: 640, newH: 360, scale: 360 / 720 });
  const tall = letterbox(720, 1280, 640);
  assert.equal(tall.newH, 640);
  assert.equal(tall.newW, 360);
});

test("similarity transform recovers a known rotation, scale and shift", () => {
  const ang = 0.3, s = 1.7, tx = 12, ty = -5;
  const a = s * Math.cos(ang), b = s * Math.sin(ang);
  const src = ARCFACE_TEMPLATE;
  const dst = src.map(([x, y]) => [a * x - b * y + tx, b * x + a * y + ty]);
  const [ea, eb, etx, ety] = estimateSimilarity(src, dst);
  close(ea, a); close(eb, b); close(etx, tx, 1e-4); close(ety, ty, 1e-4);
});

test("SCRFD decode places a box from distances around the anchor centre", () => {
  const size = 32; // strides 8,16,32 -> 4x4, 2x2, 1x1 cells, 2 anchors each
  const counts = [32, 8, 2];
  const outputs = [
    ...counts.map((n) => new Float32Array(n)),
    ...counts.map((n) => new Float32Array(n * 4)),
    ...counts.map((n) => new Float32Array(n * 10)),
  ];
  // stride 8, cell (x=1,y=2), anchor 1 -> anchor index (2*4+1)*2+1 = 19
  outputs[0][19] = 0.9;
  outputs[3].set([1, 1, 2, 2], 19 * 4);
  outputs[6].set([0, 0, 1, 0, 0.5, 0.5, 0, 1, 1, 1], 19 * 10);
  const faces = decodeScrfd(outputs, size);
  assert.equal(faces.length, 1);
  assert.deepEqual(faces[0].bbox, [8 - 8, 16 - 8, 8 + 16, 16 + 16]);
  assert.deepEqual(faces[0].kps[1], [16, 16]);
});

test("NMS keeps the best of overlapping boxes", () => {
  const faces = [
    { score: 0.6, bbox: [0, 0, 100, 100] },
    { score: 0.9, bbox: [5, 5, 105, 105] },
    { score: 0.8, bbox: [300, 300, 400, 400] },
  ];
  const kept = nms(faces);
  assert.deepEqual(kept.map((f) => f.score), [0.9, 0.8]);
  assert.ok(iou(faces[0].bbox, faces[1].bbox) > 0.4);
});

test("matching picks the best person above threshold, else Unknown", () => {
  const e = (v) => l2normalize(Float32Array.from(v));
  const gallery = { Asha: [e([1, 0, 0]), e([0.9, 0.1, 0])], Ravi: [e([0, 1, 0])] };
  const m = matchIdentity(e([0.95, 0.05, 0]), gallery, 0.45);
  assert.equal(m.name, "Asha");
  assert.ok(m.isKnown && m.similarity > 0.99);
  const u = matchIdentity(e([0, 0, 1]), gallery, 0.45);
  assert.equal(u.name, "Unknown");
  assert.equal(matchIdentity(e([1, 0, 0]), {}, 0.45).name, "Unknown");
});

test("label smoother removes single-frame flicker", () => {
  const s = new LabelSmoother(5);
  ["Asha", "Asha", "Asha"].forEach((l) => s.update(l));
  assert.equal(s.update("Unknown"), "Asha");
  assert.equal(s.update("Asha"), "Asha");
});

test("tracker keeps ids stable and drops lost tracks after the grace period", () => {
  const t = new CentroidTracker({ maxMissed: 2 });
  const [a] = t.update([[0, 0, 50, 50]]);
  const [a2] = t.update([[10, 5, 60, 55]]);
  assert.equal(a, a2);
  t.update([]); t.update([]); t.update([]);
  const [b] = t.update([[10, 5, 60, 55]]);
  assert.notEqual(b, a);
});

test("pinhole distance and zones", () => {
  close(estimateDistanceCm(86.1), 14 * 615 / 86.1);
  assert.equal(estimateDistanceCm(0), -1);
  close(estimateDistanceCm(43.05, 480), 14 * 615 / 86.1); // half-width frame, same distance
  assert.equal(distanceZone(40).zone, "close");
  assert.equal(distanceZone(80).zone, "ideal");
  assert.equal(distanceZone(500).zone, "very-far");
});

const face = (name, bbox, distanceCm = 80, isKnown = name !== "Unknown") => ({ name, isKnown, bbox, distanceCm });

test("fail-safe: unknown faces never move the robot", () => {
  const d = decide({ faces: [face("Unknown", [0, 0, 100, 100])], targetName: "Asha", frameWidth: 960 });
  assert.equal(d.command, "STOP");
  assert.equal(d.failsafe, "unknown-lockout");
});

test("fail-safe: target lost and emergency stop", () => {
  assert.equal(decide({ faces: [], targetName: "Asha", frameWidth: 960 }).failsafe, "target-lost");
  const centred = face("Asha", [430, 200, 530, 300], 200);
  assert.equal(decide({ faces: [centred], targetName: "Asha", frameWidth: 960 }).command, "FORWARD");
  const e = decide({ faces: [centred], targetName: "Asha", frameWidth: 960, emergencyStop: true });
  assert.equal(e.command, "STOP");
  assert.equal(e.failsafe, "emergency-stop");
});

test("policy: turn first, then regulate distance", () => {
  const left = face("Asha", [0, 0, 100, 100], 200);
  assert.equal(decide({ faces: [left], targetName: "Asha", frameWidth: 960 }).command, "LEFT");
  const right = face("Asha", [860, 0, 960, 100], 30);
  assert.equal(decide({ faces: [right], targetName: "Asha", frameWidth: 960 }).command, "RIGHT");
  assert.equal(decide({ faces: [face("Asha", [430, 0, 530, 100], 30)], targetName: "Asha", frameWidth: 960 }).command, "BACKWARD");
  assert.equal(decide({ faces: [face("Asha", [430, 0, 530, 100], 75)], targetName: "Asha", frameWidth: 960 }).command, "STOP");
});

test("target selection picks the largest matching face", () => {
  const small = face("Asha", [0, 0, 10, 10]);
  const big = face("Asha", [0, 0, 50, 50]);
  assert.equal(selectTarget([small, big, face("Ravi", [0, 0, 90, 90])], "Asha"), big);
});

import { cropStats, faceQuality, pca2, rollDeg, yawRatio } from "../src/vision/core.js";

test("crop stats: flat image has zero sharpness, noise has a lot", () => {
  const flat = new Uint8ClampedArray(112 * 112 * 4).fill(128);
  const s = cropStats(flat);
  close(s.brightness, 128, 1e-3);
  close(s.sharpness, 0, 1e-6);
  const noisy = flat.map((v, i) => (i % 4 === 3 ? 255 : ((i * 2654435761) >>> 24)));
  assert.ok(cropStats(noisy).sharpness > 1000);
});

test("pose from landmarks", () => {
  const frontal = [[40, 50], [72, 50], [56, 70], [44, 90], [68, 90]];
  close(yawRatio(frontal), 0);
  close(rollDeg(frontal), 0);
  const turned = [[40, 50], [72, 50], [70, 70], [44, 90], [68, 90]];
  assert.ok(yawRatio(turned) > 0.4);
  const tilted = [[40, 50], [72, 70], [56, 75], [44, 90], [68, 95]];
  assert.ok(rollDeg(tilted) > 30);
});

test("quality checklist flags small, blurry, dark and turned faces", () => {
  const good = { bbox: [0, 0, 160, 200], kps: [[40, 50], [100, 50], [70, 80], [50, 120], [90, 120]], quality: { brightness: 130, sharpness: 200 } };
  assert.ok(faceQuality(good, 960).ok);
  const bad = { ...good, bbox: [0, 0, 40, 50], quality: { brightness: 20, sharpness: 10 } };
  const q = faceQuality(bad, 960);
  assert.equal(q.ok, false);
  assert.deepEqual(q.checks.filter((c) => !c.ok).map((c) => c.key), ["size", "sharp", "light"]);
});

test("PCA separates two clusters along the first component", () => {
  const rng = (s => () => (s = (s * 16807) % 2147483647) / 2147483647)(42);
  const base = (k) => Float32Array.from({ length: 64 }, (_, j) => (j === k ? 1 : 0));
  const noisy = (v) => l2normalize(Float32Array.from(v, (x) => x + (rng() - 0.5) * 0.1));
  const a = Array.from({ length: 6 }, () => noisy(base(3)));
  const b = Array.from({ length: 6 }, () => noisy(base(40)));
  const { project, variances } = pca2([...a, ...b]);
  const xa = a.map((v) => project(v)[0]);
  const xb = b.map((v) => project(v)[0]);
  assert.ok(Math.max(...xa) < Math.min(...xb) || Math.max(...xb) < Math.min(...xa));
  assert.ok(variances[0] >= variances[1]);
  const { totalVariance } = pca2([...a, ...b]);
  assert.ok(variances[0] + variances[1] <= totalVariance + 1e-6 && variances[0] > 0.8 * totalVariance);
});
