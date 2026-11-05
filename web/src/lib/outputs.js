// Sends gesture events out of the browser, to TouchDesigner, Unreal, Physical Bridge…
//
// Browsers cannot send OSC (UDP) directly, so we send JSON over a WebSocket to the small
// bridge program in web/bridge/server.js (`npm run bridge`). The bridge then:
//   - sends the OSC messages listed in `osc` to TouchDesigner / Unreal (UDP)
//   - forwards the JSON to every other WebSocket client (TouchDesigner WebSocket DAT, Unreal…)
//   - converts it to the Physical Bridge format for clients on ws://…/physical
import { runtime } from './runtime.js';
import { createStore } from './store.js';

const KEY = 'gesture-studio-outputs';
const HAND_RATE_MS = 33; // ~30 messages per second

const defaults = {
  enabled: false,
  url: 'ws://localhost:8787',
  sendHand: true,
  mappings: {}, // gesture name -> { enabled, address }
};

function load() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) || '{}') };
  } catch {
    return { ...defaults };
  }
}

export const outputSettings = createStore(load());
outputSettings.subscribe(() => localStorage.setItem(KEY, JSON.stringify(outputSettings.get())));
export const bridgeStatus = createStore({ state: 'off', message: '' });
export const messageLog = createStore([]);

export function mappingFor(gesture) {
  return { enabled: true, address: `/gesture/${gesture}`, ...outputSettings.get().mappings[gesture] };
}

export function setMapping(gesture, patch) {
  outputSettings.set((s) => ({ ...s, mappings: { ...s.mappings, [gesture]: { ...mappingFor(gesture), ...patch } } }));
}

// ------------------------------------------------------------------ connection
let ws = null;
let retryTimer = 0;

function connect() {
  clearTimeout(retryTimer);
  const { enabled, url } = outputSettings.get();
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  if (!enabled) {
    bridgeStatus.set({ state: 'off', message: '' });
    return;
  }
  bridgeStatus.set({ state: 'connecting', message: `Connecting to ${url}…` });
  try {
    ws = new WebSocket(url);
  } catch (e) {
    bridgeStatus.set({ state: 'error', message: e.message });
    return;
  }
  ws.onopen = () => {
    bridgeStatus.set({ state: 'connected', message: `Connected to ${url}` });
    ws.send(JSON.stringify({ type: 'hello', client: 'gesture-studio-web' }));
  };
  ws.onclose = () => {
    bridgeStatus.set({ state: 'error', message: `Not connected. Is the bridge running? (npm run bridge). Retrying…` });
    retryTimer = setTimeout(connect, 2000);
  };
}

let lastUrl = null;
let lastEnabled = null;
outputSettings.subscribe(() => {
  const { url, enabled } = outputSettings.get();
  if (url !== lastUrl || enabled !== lastEnabled) {
    lastUrl = url;
    lastEnabled = enabled;
    connect();
  }
});
lastUrl = outputSettings.get().url;
lastEnabled = outputSettings.get().enabled;
connect();

function send(msg, log = true) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  if (log) {
    const line = { time: new Date().toLocaleTimeString(), sent: ws?.readyState === WebSocket.OPEN, msg };
    messageLog.set((l) => [line, ...l].slice(0, 40));
  }
}

// ------------------------------------------------------------------ message formats
/** What we send for a gesture start / end. See README "Message formats". */
export function gestureMessage({ gesture, state, confidence, mode }) {
  const map = mappingFor(gesture);
  const value = state === 'start' ? 1 : 0;
  return {
    type: 'gesture',
    gesture,
    state,
    value,
    confidence: Math.round(confidence * 1000) / 1000,
    mode,
    time: Date.now(),
    osc: [
      { address: map.address, args: [value] },
      { address: '/gesture/current', args: [state === 'start' ? gesture : 'none', state === 'start' ? confidence : 0] },
    ],
  };
}

export function handMessage({ present, x, y, size }) {
  const r = (v) => Math.round(v * 1000) / 1000;
  return {
    type: 'hand',
    present,
    x: r(x),
    y: r(y),
    size: r(size),
    osc: [
      { address: '/hand/present', args: [present ? 1 : 0] },
      { address: '/hand/x', args: [r(x)] },
      { address: '/hand/y', args: [r(y)] },
      { address: '/hand/size', args: [r(size)] },
    ],
  };
}

// ------------------------------------------------------------------ wiring
runtime.on('gesture', (ev) => {
  const s = outputSettings.get();
  if (ev.simulated || !s.enabled || !mappingFor(ev.gesture).enabled) return; // demo test buttons stay local
  send(gestureMessage(ev));
});

let lastHand = 0;
let lastPresent = false;
runtime.on('hand', (h) => {
  const s = outputSettings.get();
  if (!s.enabled || !s.sendHand) return;
  const now = performance.now();
  if (h.present === lastPresent && (!h.present || now - lastHand < HAND_RATE_MS)) return;
  lastHand = now;
  lastPresent = h.present;
  send(handMessage(h), false);
});

/** "Send test" button on the Live page. */
export function sendTest(gesture) {
  send(gestureMessage({ gesture, state: 'start', confidence: 1, mode: 'test' }));
  setTimeout(() => send(gestureMessage({ gesture, state: 'end', confidence: 1, mode: 'test' })), 300);
}
