// Feature extraction: turning MediaPipe hand landmarks into the numbers the model sees.
//
// IMPORTANT: training/gesture_studio/features.py does exactly the same maths in Python.
// Keep the two in sync (tests/features.test.js checks them against each other).
//
// MediaPipe landmarks: 21 points, each {x, y, z}
//   x, y: 0..1 across the image, z: depth relative to the wrist (about the same scale as x)
//   0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky

export const NUM_LANDMARKS = 21;
export const POSE_FEATURES = NUM_LANDMARKS * 3; // 63
export const FRAME_FEATURES = 3 + POSE_FEATURES; // 66: dx, dy, ds + hand shape
export const DEFAULT_SEQ_LEN = 16;
export const FEATURE_VERSION = 1;

// Pairs of landmarks to draw as the "skeleton".
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

/** 0..1 image coordinates -> pixel-like units, so x and y use the same scale. */
function toPixels(landmarks, width, height) {
  return landmarks.map((p) => [p.x * width, p.y * height, p.z * width]);
}

/** Wrist -> base of middle finger. Our "ruler" for hand size. */
function handSize(pts) {
  const dx = pts[9][0] - pts[0][0];
  const dy = pts[9][1] - pts[0][1];
  const dz = pts[9][2] - pts[0][2];
  return Math.max(Math.hypot(dx, dy, dz), 1e-6);
}

/**
 * 63 numbers describing the hand SHAPE, independent of position and size.
 * 1. move the wrist to (0,0,0)   -> where the hand is in the frame no longer matters
 * 2. divide by hand size          -> distance from the camera no longer matters
 * 3. mirror left hands            -> left and right hands look the same to the model
 */
export function normalizeHand(landmarks, width, height, isLeft, mirrorLeft = true) {
  const pts = toPixels(landmarks, width, height);
  const size = handSize(pts);
  const out = new Array(POSE_FEATURES);
  const flip = isLeft && mirrorLeft ? -1 : 1;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i * 3] = ((pts[i][0] - pts[0][0]) / size) * flip;
    out[i * 3 + 1] = (pts[i][1] - pts[0][1]) / size;
    out[i * 3 + 2] = (pts[i][2] - pts[0][2]) / size;
  }
  return out;
}

/** Resample rows (frames x features) captured at `times` to exactly `seqLen` evenly timed rows. */
function resample(times, rows, seqLen) {
  if (rows.length === 1) return Array.from({ length: seqLen }, () => rows[0].slice());
  const t0 = times[0];
  const tN = times[times.length - 1];
  const out = [];
  let j = 0;
  for (let i = 0; i < seqLen; i++) {
    const t = seqLen === 1 ? t0 : t0 + ((tN - t0) * i) / (seqLen - 1);
    while (j < times.length - 2 && times[j + 1] < t) j++;
    const a0 = times[j];
    const a1 = times[j + 1];
    let a = a1 === a0 ? 0 : (t - a0) / (a1 - a0);
    a = Math.min(Math.max(a, 0), 1);
    out.push(rows[j].map((v, k) => v * (1 - a) + rows[j + 1][k] * a));
  }
  return out;
}

/**
 * Features for a moving gesture. `frames`: [{t, landmarks, width, height, isLeft}]
 * Per frame: dx, dy (wrist movement since the first frame, in hand sizes, x flipped so that
 * moving to YOUR right is positive), ds (log of hand-size change: + = towards the camera),
 * then the 63 hand-shape numbers. Resampled to `seqLen` frames and flattened.
 */
export function motionFeatures(frames, seqLen = DEFAULT_SEQ_LEN, mirrorLeft = true) {
  if (!frames.length) throw new Error('motionFeatures needs at least one frame');
  let ref = null;
  const rows = [];
  const times = [];
  for (const f of frames) {
    const pts = toPixels(f.landmarks, f.width, f.height);
    const size = handSize(pts);
    if (!ref) ref = { x: pts[0][0], y: pts[0][1], size };
    const dx = -(pts[0][0] - ref.x) / ref.size;
    const dy = (pts[0][1] - ref.y) / ref.size;
    const ds = Math.log(size / ref.size);
    rows.push([dx, dy, ds, ...normalizeHand(f.landmarks, f.width, f.height, f.isLeft, mirrorLeft)]);
    times.push(f.t);
  }
  return resample(times, rows, seqLen).flat();
}

export function poseColumns() {
  const cols = [];
  for (let i = 0; i < NUM_LANDMARKS; i++) cols.push(`x${i}`, `y${i}`, `z${i}`);
  return cols;
}

export function motionColumns(seqLen = DEFAULT_SEQ_LEN) {
  const perFrame = ['dx', 'dy', 'ds', ...poseColumns()];
  const cols = [];
  for (let t = 0; t < seqLen; t++) {
    const p = `f${String(t).padStart(2, '0')}_`;
    for (const c of perFrame) cols.push(p + c);
  }
  return cols;
}
