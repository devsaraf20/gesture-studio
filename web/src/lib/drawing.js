// Canvas drawing helpers: the live skeleton over the webcam, and little sample thumbnails.
import { FRAME_FEATURES, HAND_CONNECTIONS } from './features.js';

const FINGER_OF = (i) => (i === 0 ? 0 : Math.ceil(i / 4)); // 0 wrist, 1 thumb … 5 pinky
export const FINGER_COLOURS = ['#ffffff', '#ff8a5b', '#ffd23f', '#3ddc97', '#4cc9f0', '#b388ff'];

/** Draw the hand skeleton. `landmarks` are 0..1 image coordinates. Mirrored like a selfie view. */
export function drawHand(ctx, landmarks, w, h, { mirror = true, alpha = 1, lineWidth = 3 } = {}) {
  const px = (p) => [(mirror ? 1 - p.x : p.x) * w, p.y * h];
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineCap = 'round';
  ctx.lineWidth = lineWidth;
  for (const [a, b] of HAND_CONNECTIONS) {
    const [x1, y1] = px(landmarks[a]);
    const [x2, y2] = px(landmarks[b]);
    ctx.strokeStyle = FINGER_COLOURS[FINGER_OF(b)];
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  landmarks.forEach((p, i) => {
    const [x, y] = px(p);
    ctx.fillStyle = FINGER_COLOURS[FINGER_OF(i)];
    ctx.beginPath();
    ctx.arc(x, y, i === 0 ? lineWidth * 2 : lineWidth * 1.4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

/** Draw normalised features (wrist at 0,0, hand size 1) centred in a box. */
function drawShape(ctx, shape, cx, cy, scale, lineWidth = 1.6) {
  // shape x is already "mirrored for left hands"; flip so it reads like the selfie view
  const pt = (i) => [cx - shape[i * 3] * scale, cy + shape[i * 3 + 1] * scale];
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.strokeStyle = FINGER_COLOURS[FINGER_OF(b)];
    ctx.beginPath();
    ctx.moveTo(...pt(a));
    ctx.lineTo(...pt(b));
    ctx.stroke();
  }
}

/** Thumbnail for a saved sample, so you can spot bad ones. */
export function drawSampleThumbnail(canvas, features, type) {
  const ctx = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  if (type === 'pose') {
    drawShape(ctx, features, w / 2, h * 0.8, h * 0.36);
    return;
  }
  // motion: path of the wrist + hand shape at the end
  const frames = features.length / FRAME_FEATURES;
  const path = [];
  for (let f = 0; f < frames; f++) path.push([features[f * FRAME_FEATURES], features[f * FRAME_FEATURES + 1]]);
  const xs = path.map((p) => p[0]);
  const ys = path.map((p) => p[1]);
  const span = Math.max(1.5, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const s = (w * 0.7) / span;
  const mx = (Math.max(...xs) + Math.min(...xs)) / 2;
  const my = (Math.max(...ys) + Math.min(...ys)) / 2;
  const P = ([x, y]) => [w / 2 + (x - mx) * s, h / 2 + (y - my) * s];
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach((p, i) => (i ? ctx.lineTo(...P(p)) : ctx.moveTo(...P(p))));
  ctx.stroke();
  const [ex, ey] = P(path[path.length - 1]);
  ctx.fillStyle = '#ff8a5b';
  ctx.beginPath();
  ctx.arc(ex, ey, 3.5, 0, Math.PI * 2);
  ctx.fill();
  const last = features.slice((frames - 1) * FRAME_FEATURES + 3, frames * FRAME_FEATURES);
  ctx.globalAlpha = 0.5;
  drawShape(ctx, last, w * 0.82, h * 0.92, h * 0.12, 1);
  ctx.globalAlpha = 1;
}
