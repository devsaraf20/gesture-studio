// Checks that the browser's feature maths matches the Python training code exactly.
// Fixture made by: python training/tests/make_feature_fixture.py
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { motionFeatures, normalizeHand } from '../src/lib/features.js';

const fx = JSON.parse(readFileSync(new URL('./fixtures/features_fixture.json', import.meta.url)));
const toObj = (L) => L.map(([x, y, z]) => ({ x, y, z }));
const close = (a, b) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-6, `index ${i}: ${v} vs ${b[i]}`));
};

test('pose features match Python', () => {
  for (const p of fx.poses) close(normalizeHand(toObj(p.landmarks), p.width, p.height, p.is_left), p.expected);
});

test('motion features match Python', () => {
  const frames = fx.motion.frames.map((f) => ({ ...f, landmarks: toObj(f.landmarks), isLeft: f.is_left }));
  close(motionFeatures(frames, fx.motion.seq_len), fx.motion.expected);
});
