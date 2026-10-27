/**
 * Tracking, distance and robot-following policy: JavaScript ports of
 * app/core/tracking_utils.py, distance_utils.py and robot_utils.py, so the
 * browser demo obeys exactly the same rules as the desktop app and robot.
 */

export const CONFIG = {
  similarityThreshold: 0.45,
  smoothingWindow: 5,
  knownFaceWidthCm: 14.0,
  focalLengthPx: 615.0, // calibrated for a 960px-wide frame; scaled to the real width below
  referenceWidthPx: 960,
  tooCloseCm: 50,
  idealMaxCm: 100,
  tooFarCm: 150,
  deadZonePx: 60, // at reference width; scaled to the real width below
  forwardCm: 100,
  backwardCm: 50,
};

/** Rolling majority vote over the last `window` labels (LabelSmoother). */
export class LabelSmoother {
  constructor(window = CONFIG.smoothingWindow) {
    this.window = window;
    this.history = [];
  }
  update(label) {
    this.history.push(label);
    if (this.history.length > this.window) this.history.shift();
    const counts = new Map();
    for (const l of this.history) counts.set(l, (counts.get(l) || 0) + 1);
    let best = label, bestCount = -1;
    // Ties go to the most recent label, so a new identity can take over.
    for (let i = this.history.length - 1; i >= 0; i--) {
      const l = this.history[i];
      if (counts.get(l) > bestCount) { best = l; bestCount = counts.get(l); }
    }
    return best;
  }
}

const centroid = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Nearest-centroid tracker with a distance gate and a missed-frame grace period. */
export class CentroidTracker {
  constructor({ maxDistancePx = 120, maxMissed = 10, smoothingWindow = CONFIG.smoothingWindow } = {}) {
    this.maxDistance = maxDistancePx;
    this.maxMissed = maxMissed;
    this.smoothingWindow = smoothingWindow;
    this.tracks = new Map();
    this.nextId = 1;
  }

  /** Returns, for each detection index, the track id it was assigned to. */
  update(bboxes) {
    const cents = bboxes.map(centroid);
    const unmatchedDet = new Set(bboxes.map((_, i) => i));
    const unmatchedTracks = new Set(this.tracks.keys());
    const assignment = new Array(bboxes.length).fill(-1);
    const pairs = [];
    for (const [id, t] of this.tracks) {
      cents.forEach((c, i) => {
        const d = dist(t.centroid, c);
        if (d <= this.maxDistance) pairs.push([d, id, i]);
      });
    }
    pairs.sort((p, q) => p[0] - q[0]);
    for (const [, id, i] of pairs) {
      if (!unmatchedTracks.has(id) || !unmatchedDet.has(i)) continue;
      const t = this.tracks.get(id);
      t.bbox = bboxes[i];
      t.centroid = cents[i];
      t.missed = 0;
      unmatchedTracks.delete(id);
      unmatchedDet.delete(i);
      assignment[i] = id;
    }
    for (const id of unmatchedTracks) this.tracks.get(id).missed += 1;
    for (const [id, t] of [...this.tracks]) if (t.missed > this.maxMissed) this.tracks.delete(id);
    for (const i of unmatchedDet) {
      const id = this.nextId++;
      this.tracks.set(id, { bbox: bboxes[i], centroid: cents[i], missed: 0, smoother: new LabelSmoother(this.smoothingWindow) });
      assignment[i] = id;
    }
    return assignment;
  }

  smooth(trackId, label) {
    const t = this.tracks.get(trackId);
    return t ? t.smoother.update(label) : label;
  }

  reset() {
    this.tracks.clear();
  }
}

/** Pinhole model: distance = real width × focal length / pixel width. */
export function estimateDistanceCm(faceWidthPx, frameWidthPx = CONFIG.referenceWidthPx) {
  if (faceWidthPx <= 0) return -1;
  const focal = CONFIG.focalLengthPx * (frameWidthPx / CONFIG.referenceWidthPx);
  return (CONFIG.knownFaceWidthCm * focal) / faceWidthPx;
}

export function distanceZone(cm) {
  if (cm < 0) return { zone: "unknown", label: "Unknown" };
  if (cm <= CONFIG.tooCloseCm) return { zone: "close", label: "Too close" };
  if (cm <= CONFIG.idealMaxCm) return { zone: "ideal", label: "Ideal range" };
  if (cm <= CONFIG.tooFarCm) return { zone: "far", label: "A bit far" };
  return { zone: "very-far", label: "Too far" };
}

/** Largest known face whose name is the chosen target (select_recognized_target). */
export function selectTarget(faces, targetName) {
  const matches = faces.filter((f) => f.isKnown && f.name === targetName);
  if (!matches.length) return null;
  const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
  return matches.reduce((best, f) => (area(f.bbox) > area(best.bbox) ? f : best));
}

/**
 * The robot's decision for one frame, with every fail-safe from the
 * desktop app. Returns { command, reason, failsafe }.
 */
export function decide({ faces, targetName, frameWidth, emergencyStop = false }) {
  if (emergencyStop) {
    return { command: "STOP", reason: "Emergency stop is engaged", failsafe: "emergency-stop" };
  }
  if (!targetName) {
    return { command: "STOP", reason: "No target chosen", failsafe: "no-target" };
  }
  const target = selectTarget(faces, targetName);
  if (!target) {
    const unknownVisible = faces.some((f) => !f.isKnown);
    return unknownVisible
      ? { command: "STOP", reason: "Only unknown faces in view. Unknown people can never move the robot", failsafe: "unknown-lockout" }
      : { command: "STOP", reason: `${targetName} is not in view. Holding position`, failsafe: "target-lost" };
  }
  const scale = frameWidth / CONFIG.referenceWidthPx;
  const offset = (target.bbox[0] + target.bbox[2]) / 2 - frameWidth / 2;
  if (Math.abs(offset) > CONFIG.deadZonePx * scale) {
    return offset < 0
      ? { command: "LEFT", reason: `${targetName} is ${Math.round(-offset / scale)} px left of centre`, failsafe: null }
      : { command: "RIGHT", reason: `${targetName} is ${Math.round(offset / scale)} px right of centre`, failsafe: null };
  }
  const cm = target.distanceCm;
  if (cm == null || cm < 0) return { command: "STOP", reason: "Distance unknown. Holding position", failsafe: "no-distance" };
  if (cm > CONFIG.forwardCm) return { command: "FORWARD", reason: `${targetName} is ${Math.round(cm)} cm away. Closing in`, failsafe: null };
  if (cm < CONFIG.backwardCm) return { command: "BACKWARD", reason: `${targetName} is ${Math.round(cm)} cm away. Backing off`, failsafe: null };
  return { command: "STOP", reason: `${targetName} is centred at ${Math.round(cm)} cm. Holding`, failsafe: null };
}
