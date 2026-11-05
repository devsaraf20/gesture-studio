// The live gesture engine, shared by the Live and Demo pages.
//
// camera frame -> features -> model -> probabilities -> smoother -> gesture events
//
// Events you can listen to with runtime.on(type, fn):
//   'prediction' { modelId, raw, smoothed, top, topProb, active, handVisible }
//   'gesture'    { gesture, state: 'start'|'end', confidence, mode, modelName }
//   'hand'       { present, x, y, size }   (x, y: 0..1 as seen in the mirrored view)
import { motionFeatures, normalizeHand } from './features.js';
import { onFrame } from './handTracker.js';
import { GestureSmoother } from './smoother.js';
import { createEmitter, createStore } from './store.js';

const SETTINGS_KEY = 'gesture-studio-live-settings';

export const DEFAULT_SETTINGS = {
  threshold: 0.8,
  smoothing: 0.6,
  holdMs: 120,
  cooldownMs: 300,
  motionCooldownMs: 900,
};

function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

/** settings: thresholds etc. models: [{ model, smoother, status }] */
export const liveSettings = createStore(loadSettings());
liveSettings.subscribe(() => localStorage.setItem(SETTINGS_KEY, JSON.stringify(liveSettings.get())));
export const loadedModels = createStore([]);

const emitter = createEmitter();
export const runtime = { on: emitter.on };

const motionBuffers = new Map(); // modelId -> recent frames

export function addModel(model) {
  const smoother = new GestureSmoother(model.meta.classes, model.meta.mode);
  loadedModels.set((list) => [...list, { model, smoother }]);
}

export function removeModel(id) {
  const entry = loadedModels.get().find((m) => m.model.id === id);
  if (!entry) return;
  if (entry.smoother.active) {
    emitter.emit('gesture', { gesture: entry.smoother.active, state: 'end', confidence: 0, mode: entry.model.meta.mode, modelName: entry.model.meta.name });
  }
  entry.model.dispose();
  motionBuffers.delete(id);
  loadedModels.set((list) => list.filter((m) => m.model.id !== id));
}

/** For the demo page's test buttons: pretend a gesture happened. */
export function simulateGesture(gesture, state = 'start') {
  emitter.emit('gesture', { gesture, state, confidence: 1, mode: 'pose', modelName: 'test', simulated: true });
}

function motionWindow(meta) {
  return Math.min(Math.max(meta.window_ms || 900, 300), 3000);
}

onFrame((frame) => {
  const { hand, width, height, t } = frame;
  const settings = liveSettings.get();

  if (hand) {
    // hand position for continuous controls (mirrored so it matches what you see)
    const w = hand.landmarks[0];
    const m = hand.landmarks[9];
    emitter.emit('hand', {
      present: true,
      x: 1 - (w.x + m.x) / 2,
      y: (w.y + m.y) / 2,
      size: Math.hypot((m.x - w.x) * width, (m.y - w.y) * height) / height,
    });
  } else {
    emitter.emit('hand', { present: false, x: 0.5, y: 0.5, size: 0 });
  }

  for (const { model, smoother } of loadedModels.get()) {
    const { meta } = model;
    let raw = null;
    if (meta.mode === 'motion') {
      const buf = motionBuffers.get(model.id) || [];
      if (hand) buf.push({ t, landmarks: hand.landmarks, width, height, isLeft: hand.isLeft });
      else buf.length = 0; // hand lost: start again
      const windowMs = motionWindow(meta);
      while (buf.length && t - buf[0].t > windowMs) buf.shift();
      motionBuffers.set(model.id, buf);
      // only predict once we have (almost) a full window of movement
      if (buf.length >= 4 && buf[buf.length - 1].t - buf[0].t >= windowMs * 0.8) {
        raw = model.predict(motionFeatures(buf, meta.seq_len));
      }
    } else if (hand) {
      raw = model.predict(normalizeHand(hand.landmarks, width, height, hand.isLeft, meta.mirror_left_hands !== false));
    }

    const result = smoother.update(raw, t, {
      ...settings,
      cooldownMs: meta.mode === 'motion' ? settings.motionCooldownMs : settings.cooldownMs,
    });
    if (smoother.wantsBufferReset) {
      motionBuffers.set(model.id, []);
      smoother.wantsBufferReset = false;
    }
    emitter.emit('prediction', { modelId: model.id, raw, handVisible: !!hand, ...result });
    for (const ev of result.events) emitter.emit('gesture', { ...ev, mode: meta.mode, modelName: meta.name });
  }
});
