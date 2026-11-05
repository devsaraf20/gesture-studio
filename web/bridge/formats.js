// Message formats the bridge can produce from Gesture Studio's own JSON.
//
// PHYSICAL BRIDGE FORMAT (v1)
// ---------------------------
// The physical-bridge repository did not define a message format yet, so this is our proposal.
// Everything about it lives in this one function: change it here if your Physical Bridge
// expects something different. Each message is one JSON object:
//
//   { "pb": 1,                       format version
//     "source": "gesture-studio",    who sent it
//     "kind": "trigger" | "vector",  trigger = on/off event, vector = continuous numbers
//     "channel": "gesture/peace",    what it is about (like an OSC address without the first /)
//     "value": 1,                    trigger: 1 = started, 0 = ended | vector: [x, y, size]
//     "confidence": 0.93,            triggers only
//     "ts": 1730800000000 }          milliseconds since 1970 (Date.now())

export function toPhysicalBridge(msg) {
  const base = { pb: 1, source: 'gesture-studio', ts: msg.time ?? Date.now() };
  if (msg.type === 'gesture') {
    return { ...base, kind: 'trigger', channel: `gesture/${msg.gesture}`, value: msg.value, confidence: msg.confidence };
  }
  if (msg.type === 'hand') {
    return { ...base, kind: 'vector', channel: 'hand', value: msg.present ? [msg.x, msg.y, msg.size] : null };
  }
  return null; // other message types are not forwarded
}

/** The plain JSON sent to normal WebSocket clients: our message minus the OSC instructions. */
export function toPlainJson(msg) {
  const { osc, ...rest } = msg;
  return rest;
}
