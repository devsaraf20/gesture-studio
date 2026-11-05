// End-to-end test of the bridge: browser-style WebSocket message in -> OSC (UDP),
// plain JSON and Physical Bridge JSON out.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import dgram from 'node:dgram';
import { WebSocket } from 'ws';
import { decodeOsc, encodeOsc } from '../bridge/osc.js';

test('OSC encode/decode round trip', () => {
  const { address, args } = decodeOsc(encodeOsc('/gesture/peace', [1, 'peace', 0.5]));
  assert.equal(address, '/gesture/peace');
  assert.deepEqual(args, [1, 'peace', 0.5]);
  assert.equal(encodeOsc('/a', [1]).length % 4, 0);
});

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const open = (url) => new Promise((res, rej) => { const ws = new WebSocket(url); ws.on('open', () => res(ws)); ws.on('error', rej); });

test('bridge forwards a gesture to OSC, JSON and Physical Bridge clients', async () => {
  const port = 18787;
  const proc = spawn(process.execPath, ['bridge/server.js'], { env: { ...process.env, BRIDGE_PORT: String(port) }, stdio: 'pipe' });
  const udp = dgram.createSocket('udp4');
  const oscMessages = [];
  udp.on('message', (b) => oscMessages.push(decodeOsc(b)));
  await new Promise((r) => udp.bind(10000, r));
  try {
    for (let i = 0; i < 50; i++) { try { await (await open(`ws://localhost:${port}`)).close(); break; } catch { await wait(100); } }
    const sender = await open(`ws://localhost:${port}`);
    const json = await open(`ws://localhost:${port}`);
    const physical = await open(`ws://localhost:${port}/physical`);
    const got = { json: null, physical: null };
    json.on('message', (d) => (got.json = JSON.parse(d)));
    physical.on('message', (d) => (got.physical = JSON.parse(d)));
    sender.send(JSON.stringify({ type: 'hello' }));
    sender.send(JSON.stringify({
      type: 'gesture', gesture: 'peace', state: 'start', value: 1, confidence: 0.9, mode: 'pose', time: 123,
      osc: [{ address: '/gesture/peace', args: [1] }, { address: '/gesture/current', args: ['peace', 0.9] }],
    }));
    await wait(300);
    assert.equal(got.json.gesture, 'peace');
    assert.equal(got.json.osc, undefined);
    assert.deepEqual(got.physical, { pb: 1, source: 'gesture-studio', ts: 123, kind: 'trigger', channel: 'gesture/peace', value: 1, confidence: 0.9 });
    assert.deepEqual(oscMessages.map((m) => m.address), ['/gesture/peace', '/gesture/current']);
    assert.equal(oscMessages[0].args[0], 1);
    [sender, json, physical].forEach((w) => w.close());
  } finally {
    udp.close();
    proc.kill();
  }
});
