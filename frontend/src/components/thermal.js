// Thermal colour scale: cold indigo -> magenta -> orange -> hot yellow.
// Used everywhere a number becomes colour: known faces run "hot",
// unknown faces stay "cold".
const STOPS = [
  [0.0, [43, 28, 107]],
  [0.35, [180, 35, 122]],
  [0.7, [242, 84, 45]],
  [1.0, [255, 209, 92]],
];

export function thermal(t, alpha = 1) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, c1] = STOPS[i];
    const [p0, c0] = STOPS[i - 1];
    if (x <= p1) {
      const k = (x - p0) / (p1 - p0);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
      return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
    }
  }
  return `rgba(255,209,92,${alpha})`;
}

export const COLD = "#B9ACE6";
export const HOT = "#FF8A3D";
