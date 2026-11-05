// Gesture Studio bridge: the "post office" between the browser and your creative software.
//
//   npm run bridge
//
//   browser (Live page) --WebSocket JSON--> bridge --OSC over UDP--> TouchDesigner, Unreal
//                                                  --WebSocket JSON-->  any client on ws://localhost:8787
//                                                  --Physical Bridge--> clients on ws://localhost:8787/physical
//                                                                       (+ optional forward URL)
// Settings: bridge/config.json (ports, OSC targets).
import dgram from 'node:dgram';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import { WebSocket, WebSocketServer } from 'ws';
import { toPhysicalBridge, toPlainJson } from './formats.js';
import { encodeOsc } from './osc.js';

const config = JSON.parse(readFileSync(new URL('./config.json', import.meta.url), 'utf8'));
const PORT = Number(process.env.BRIDGE_PORT || config.websocketPort || 8787);
const targets = (config.oscTargets || []).filter((t) => t.enabled !== false);
const udp = dgram.createSocket('udp4');
const stats = { messages: 0, osc: 0, since: new Date() };

// ------------------------------------------------------------------ HTTP status page
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
  res.end(
    [
      'Gesture Studio bridge is running.',
      '',
      `WebSocket (JSON):            ws://localhost:${PORT}`,
      `WebSocket (Physical Bridge): ws://localhost:${PORT}/physical`,
      `Connected clients: ${wss.clients.size}`,
      `Messages received: ${stats.messages}, OSC packets sent: ${stats.osc}`,
      '',
      'OSC targets:',
      ...targets.map((t) => `  ${t.name}: ${t.host}:${t.port}`),
    ].join('\n'),
  );
});

// ------------------------------------------------------------------ WebSocket hub
const wss = new WebSocketServer({ server });

wss.on('connection', (socket, req) => {
  socket.kind = req.url?.startsWith('/physical') ? 'physical' : 'json';
  const who = `${req.socket.remoteAddress} (${socket.kind})`;
  console.log(`+ client connected: ${who}  [${wss.clients.size} total]`);
  socket.on('close', () => console.log(`- client left: ${who}`));
  socket.on('message', (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return console.warn('ignored a message that was not JSON');
    }
    if (msg.type === 'hello') {
      socket.isSender = true;
      return;
    }
    handle(msg, socket);
  });
});

function handle(msg, from) {
  stats.messages++;
  // 1. OSC out
  for (const { address, args } of msg.osc || []) {
    let packet;
    try {
      packet = encodeOsc(address, args);
    } catch (e) {
      console.warn(e.message);
      continue;
    }
    for (const t of targets) {
      udp.send(packet, t.port, t.host);
      stats.osc++;
    }
  }
  // 2. WebSocket out
  const plain = JSON.stringify(toPlainJson(msg));
  const pbMsg = toPhysicalBridge(msg);
  const pb = pbMsg && JSON.stringify(pbMsg);
  for (const client of wss.clients) {
    if (client === from || client.readyState !== WebSocket.OPEN) continue;
    if (client.kind === 'physical') {
      if (pb) client.send(pb);
    } else client.send(plain);
  }
  if (pb) forwardPhysical(pb);
  if (msg.type === 'gesture') {
    console.log(`  ${msg.state === 'start' ? '▶' : '■'} ${msg.gesture} ${msg.state} (${Math.round((msg.confidence || 0) * 100)}%)`);
  }
}

// ------------------------------------------------------------------ optional: forward to a Physical Bridge server
let forward = null;
function connectForward() {
  const url = config.physicalBridge?.forwardUrl;
  if (!url) return;
  forward = new WebSocket(url);
  forward.on('open', () => console.log(`Forwarding Physical Bridge messages to ${url}`));
  forward.on('error', () => {});
  forward.on('close', () => setTimeout(connectForward, 3000));
}
function forwardPhysical(text) {
  if (forward?.readyState === WebSocket.OPEN) forward.send(text);
}

server.listen(PORT, () => {
  console.log(`\nGesture Studio bridge running`);
  console.log(`  WebSocket JSON:            ws://localhost:${PORT}`);
  console.log(`  WebSocket Physical Bridge: ws://localhost:${PORT}/physical`);
  for (const t of targets) console.log(`  OSC -> ${t.name.padEnd(14)} ${t.host}:${t.port}`);
  console.log('Waiting for the Live page to connect (tick "Send to bridge")…\n');
  connectForward();
});
server.on('error', (e) => {
  if (e.code === 'EADDRINUSE') console.error(`Port ${PORT} is already in use. Is the bridge already running?`);
  else console.error(e);
  process.exit(1);
});
