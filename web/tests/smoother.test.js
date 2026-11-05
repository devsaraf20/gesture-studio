// The smoother turns noisy per-frame predictions into clean start/end events.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GestureSmoother } from '../src/lib/smoother.js';

const settings = { smoothing: 0.5, threshold: 0.8, holdMs: 100, cooldownMs: 0 };
const run = (sm, probs, from, to, step = 33) => {
  const events = [];
  for (let t = from; t <= to; t += step) events.push(...sm.update(probs, t, settings).events);
  return events;
};

test('a pose starts after the hold time and ends when the hand goes away', () => {
  const sm = new GestureSmoother(['fist', 'peace', 'none'], 'pose');
  const start = run(sm, [0.05, 0.9, 0.05], 0, 500);
  assert.deepEqual(start.map((e) => [e.gesture, e.state]), [['peace', 'start']]);
  const end = run(sm, null, 533, 1200);
  assert.deepEqual(end.map((e) => [e.gesture, e.state]), [['peace', 'end']]);
});

test('a single flickering frame does not trigger anything', () => {
  const sm = new GestureSmoother(['fist', 'peace', 'none'], 'pose');
  run(sm, [0.05, 0.05, 0.9], 0, 300);
  const ev = [...sm.update([0.99, 0, 0.01], 333, settings).events, ...run(sm, [0.05, 0.05, 0.9], 366, 700)];
  assert.equal(ev.length, 0);
});

test('rest classes ("none") never trigger', () => {
  const sm = new GestureSmoother(['fist', 'none'], 'pose');
  assert.equal(run(sm, [0, 1], 0, 1000).length, 0);
});

test('a motion fires once, then switches off', () => {
  const sm = new GestureSmoother(['idle', 'swipe_left'], 'motion');
  // even with cooldown 0, a motion can't fire again within 500 ms of the last one
  const ev = run(sm, [0.02, 0.98], 0, 600);
  assert.deepEqual(ev.map((e) => [e.gesture, e.state]), [['swipe_left', 'start'], ['swipe_left', 'end']]);
});
