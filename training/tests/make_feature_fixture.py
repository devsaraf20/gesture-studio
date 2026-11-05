"""Writes web/tests/fixtures/features_fixture.json: raw landmarks + the features Python computes.

The web test (web/tests/features.test.js) recomputes them in JavaScript and checks they match.
Run this again whenever you change gesture_studio/features.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gesture_studio.features import motion_features, normalize_hand  # noqa: E402

rng = np.random.default_rng(0)


def lm():
    base = rng.uniform(0.3, 0.7, 3) * np.array([1, 1, 0])
    return (base + rng.normal(0, 0.08, (21, 3))).round(5)


poses = []
for is_left in (False, True):
    L = lm()
    poses.append({"landmarks": L.tolist(), "width": 640, "height": 480, "is_left": is_left,
                  "expected": normalize_hand(L, 640, 480, is_left).tolist()})

frames, t = [], 0.0
for i in range(23):
    t += float(rng.uniform(25, 45))
    L = lm() + np.array([0.01 * i, -0.005 * i, 0])
    frames.append({"t": round(t, 3), "landmarks": L.round(5).tolist(), "width": 1280, "height": 720, "is_left": i % 2 == 0})
motion = {"frames": frames, "seq_len": 16, "expected": motion_features(frames, 16).tolist()}

out = os.path.join(HERE, "..", "..", "web", "tests", "fixtures", "features_fixture.json")
with open(out, "w") as f:
    json.dump({"poses": poses, "motion": motion}, f)
print("wrote", os.path.normpath(out))
