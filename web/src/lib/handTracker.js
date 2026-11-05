// Webcam + MediaPipe Hands. One shared tracker for the whole app, so switching pages
// doesn't restart the camera.
//
// Every video frame we emit:
//   { t, width, height, hand: { landmarks: [{x,y,z} x21], isLeft, score } | null }
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { createEmitter, createStore } from './store.js';

const WASM_PATH = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const LOCAL_MODEL = `${import.meta.env.BASE_URL}mediapipe/hand_landmarker.task`;
const REMOTE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export const trackerStatus = createStore({ state: 'off', message: '', fps: 0 });
const emitter = createEmitter();
let landmarker = null;
let video = null;
let stream = null;
let rafId = 0;
let lastVideoTime = -1;
let lastFrame = null;

export const onFrame = (fn) => emitter.on('frame', fn);

// For automated tests without a webcam: feed in a fake frame (only in `npm run dev`).
if (import.meta.env.DEV) window.__gestureStudioInjectFrame = (frame) => emitter.emit('frame', frame);
export const getLastFrame = () => lastFrame;
export const getVideo = () => video;

async function pickModelPath() {
  // Use a local copy if you've put one in web/public/mediapipe/ (handy for offline workshops).
  try {
    const r = await fetch(LOCAL_MODEL, { method: 'HEAD' });
    if (r.ok && !(r.headers.get('content-type') || '').includes('html')) return LOCAL_MODEL;
  } catch {
    /* fall through */
  }
  return REMOTE_MODEL;
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  const modelAssetPath = await pickModelPath();
  const options = (delegate) => ({
    baseOptions: { modelAssetPath, delegate },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(vision, options('GPU'));
  } catch (e) {
    console.warn('GPU hand tracking unavailable, using CPU', e);
    return HandLandmarker.createFromOptions(vision, options('CPU'));
  }
}

export async function startCamera() {
  if (trackerStatus.get().state === 'running' || trackerStatus.get().state === 'loading') return;
  try {
    trackerStatus.set({ state: 'loading', message: 'Asking for camera permission…', fps: 0 });
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
      audio: false,
    });
    video = document.createElement('video');
    video.playsInline = true;
    video.muted = true;
    video.srcObject = stream;
    await video.play();
    if (!landmarker) {
      trackerStatus.set({ state: 'loading', message: 'Loading the hand tracking model…', fps: 0 });
      landmarker = await createLandmarker();
    }
    trackerStatus.set({ state: 'running', message: '', fps: 0 });
    loop();
  } catch (e) {
    console.error(e);
    const message =
      e.name === 'NotAllowedError'
        ? 'Camera permission was blocked. Click the camera icon in the address bar to allow it.'
        : e.name === 'NotFoundError'
          ? 'No webcam found.'
          : `Could not start: ${e.message}`;
    trackerStatus.set({ state: 'error', message, fps: 0 });
    stopCamera(false);
  }
}

export function stopCamera(updateStatus = true) {
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video = null;
  lastFrame = null;
  if (updateStatus) trackerStatus.set({ state: 'off', message: '', fps: 0 });
}

let fpsCount = 0;
let fpsSince = performance.now();

function loop() {
  rafId = requestAnimationFrame(loop);
  if (!video || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const now = performance.now();
  let hand = null;
  try {
    const res = landmarker.detectForVideo(video, now);
    if (res.landmarks?.length) {
      const h = res.handedness?.[0]?.[0];
      hand = { landmarks: res.landmarks[0], isLeft: h?.categoryName === 'Left', score: h?.score ?? 1 };
    }
  } catch (e) {
    console.error(e);
  }
  lastFrame = { t: now, width: video.videoWidth, height: video.videoHeight, hand };
  emitter.emit('frame', lastFrame);

  fpsCount++;
  if (now - fpsSince > 1000) {
    trackerStatus.set((s) => ({ ...s, fps: Math.round((fpsCount * 1000) / (now - fpsSince)) }));
    fpsCount = 0;
    fpsSince = now;
  }
}
