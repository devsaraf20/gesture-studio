"""
Generates the small SAMPLE datasets in sample_data/ so you can test training without a webcam.

The hands are SYNTHETIC: a simple 3D stick-figure hand with bendable fingers, randomly
rotated, scaled and moved around the frame, then turned into features with exactly the same
normalisation the web app uses. Real webcam data is noisier - always train on your own data
for real projects.

    python make_sample_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from gesture_studio.features import (DEFAULT_SEQ_LEN, motion_columns, motion_features,
                                     normalize_hand, pose_columns)

W, H = 640, 480
HERE = os.path.dirname(os.path.abspath(__file__))

# Finger base positions relative to the wrist (units: wrist->middle-knuckle = 1). y points down.
BASES = {
    "thumb": (0.35, -0.25, 0.0),
    "index": (0.28, -0.95, 0.0),
    "middle": (0.0, -1.0, 0.0),
    "ring": (-0.22, -0.92, 0.0),
    "pinky": (-0.42, -0.80, 0.0),
}
SEGMENTS = {
    "thumb": (0.40, 0.32, 0.28),
    "index": (0.45, 0.27, 0.22),
    "middle": (0.50, 0.30, 0.24),
    "ring": (0.46, 0.28, 0.22),
    "pinky": (0.36, 0.22, 0.20),
}
SPREAD = {"index": 8, "middle": 0, "ring": -7, "pinky": -15}  # degrees, sideways tilt
EXT, CURL = (5, 5, 5), (85, 95, 60)

# per finger: curl angles for the 3 joints. Thumb: (direction angle, curl angles)
POSES = {
    "open_palm": {"fingers": {f: EXT for f in SPREAD}, "spread": 1.6, "thumb": (50, (5, 5, 5))},
    "fist": {"fingers": {f: CURL for f in SPREAD}, "spread": 1.0, "thumb": (-20, (35, 45, 30))},
    "point": {"fingers": {"index": EXT, "middle": CURL, "ring": CURL, "pinky": CURL},
              "spread": 1.0, "thumb": (-20, (35, 45, 30))},
    "peace": {"fingers": {"index": EXT, "middle": EXT, "ring": CURL, "pinky": CURL},
              "spread": 2.2, "thumb": (-20, (35, 45, 30))},
    "thumbs_up": {"fingers": {f: CURL for f in SPREAD}, "spread": 1.0, "thumb": (10, (0, 0, 0))},
}


def rot(roll, yaw, pitch):
    r, y, p = np.radians([roll, yaw, pitch])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rz @ Ry @ Rx


def build_hand(pose, rng, jitter_deg):
    """Return 21 landmarks in hand space (wrist at origin)."""
    pts = [np.zeros(3)]
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        base = np.array(BASES[finger], dtype=float)
        if finger == "thumb":
            angle, curls = pose["thumb"]
            angle = np.radians(angle + rng.normal(0, jitter_deg))
            side = np.array([np.sin(angle), -np.cos(angle)])
            pts.append(base)
            p, c = base.copy(), 0.0
            for seg, curl in zip(SEGMENTS[finger], curls):
                c += np.radians(curl + rng.normal(0, jitter_deg))
                d = np.array([side[0] * np.cos(c) - 0.6 * np.sin(c), side[1] * np.cos(c), -np.sin(c) * 0.8])
                p = p + seg * d / np.linalg.norm(d)
                pts.append(p.copy())
        else:
            spread = np.radians(SPREAD[finger] * pose["spread"] + rng.normal(0, jitter_deg / 2))
            p, c = base.copy(), 0.0
            pts.append(p.copy())
            for seg, curl in zip(SEGMENTS[finger][:3], pose["fingers"][finger]):
                c += np.radians(curl + rng.normal(0, jitter_deg))
                d = np.array([np.sin(spread) * np.cos(c), -np.cos(spread) * np.cos(c), -np.sin(c)])
                p = p + seg * d
                pts.append(p.copy())
    # thumb has CMC + 3 joints = 4 points, other fingers MCP + 3 = 4 points -> 1 + 20 = 21
    return np.array(pts)


def to_image(hand, center, size_px, R, is_left, rng, noise):
    h = hand @ R.T
    h = h + rng.normal(0, noise, h.shape)
    if is_left:
        h[:, 0] *= -1
    x = (center[0] + h[:, 0] * size_px) / W
    y = (center[1] + h[:, 1] * size_px) / H
    z = h[:, 2] * size_px / W
    return np.stack([x, y, z], 1)


def random_take(rng):
    return dict(
        roll=rng.uniform(-35, 35), yaw=rng.uniform(-55, 55), pitch=rng.uniform(-40, 40),
        size=rng.uniform(55, 150), center=np.array([rng.uniform(150, 490), rng.uniform(150, 380)]),
        is_left=rng.random() < 0.4, sloppy=rng.random() < 0.3,
    )


def make_poses(rng, takes_per_class=10, samples_per_take=16):
    rows, take_id = [], 0
    for label, pose in POSES.items():
        for _ in range(takes_per_class):
            take_id += 1
            tk = random_take(rng)
            take_rng = np.random.default_rng(rng.integers(1 << 31))
            hand_base_seed = take_rng.integers(1 << 31)
            for _ in range(samples_per_take):
                # same basic hand for the whole take, small wobble between samples
                hand = build_hand(pose, np.random.default_rng(hand_base_seed), 8 if not tk["sloppy"] else 16)
                hand += take_rng.normal(0, 0.02, hand.shape)
                R = rot(tk["roll"] + take_rng.normal(0, 3), tk["yaw"] + take_rng.normal(0, 3),
                        tk["pitch"] + take_rng.normal(0, 3))
                center = tk["center"] + take_rng.normal(0, 8, 2)
                lm = to_image(hand, center, tk["size"] * take_rng.uniform(0.95, 1.05), R, tk["is_left"],
                              take_rng, 0.012)
                feats = normalize_hand(lm, W, H, tk["is_left"])
                rows.append([label, take_id] + list(np.round(feats, 4)))
    return pd.DataFrame(rows, columns=["label", "take"] + pose_columns())


def ease(t):
    return t * t * (3 - 2 * t)


def make_motions(rng, per_class=40):
    rows, take_id = [], 1000
    for label in ["swipe_left", "swipe_right", "swipe_up", "idle"]:
        for _ in range(per_class):
            take_id += 1
            tk = random_take(rng)
            duration = rng.uniform(550, 1100)
            n = max(8, int(duration / 33))
            times = np.sort(np.concatenate([[0, duration], rng.uniform(0, duration, n - 2)]))
            dist = rng.uniform(2.5, 5.5) * tk["size"]
            pad = rng.uniform(0, 0.2)
            pose_name = "open_palm" if label != "idle" or rng.random() < 0.5 else rng.choice(list(POSES))
            hand = build_hand(POSES[pose_name], rng, 10)
            frames = []
            start = tk["center"].copy()
            # user's right = image left (the camera is not mirrored)
            direction = {"swipe_left": (1, 0), "swipe_right": (-1, 0), "swipe_up": (0, -1), "idle": (0, 0)}[label]
            drift = rng.normal(0, 0.25, 2) * tk["size"]
            wobble = rng.normal(0, 0.15) * tk["size"]
            for t in times:
                u = np.clip((t / duration - pad) / (1 - 2 * pad), 0, 1)
                e = ease(u)
                offset = np.array(direction) * dist * e + drift * u
                offset += np.array(direction[::-1]) * wobble * np.sin(np.pi * e)
                R = rot(tk["roll"] + 10 * e * direction[0], tk["yaw"], tk["pitch"])
                lm = to_image(hand, start + offset, tk["size"], R, tk["is_left"], rng, 0.015)
                frames.append({"t": t, "landmarks": lm, "width": W, "height": H, "is_left": tk["is_left"]})
            feats = motion_features(frames, DEFAULT_SEQ_LEN)
            rows.append([label, take_id, round(duration)] + list(np.round(feats, 4)))
    return pd.DataFrame(rows, columns=["label", "take", "duration_ms"] + motion_columns(DEFAULT_SEQ_LEN))


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    out = os.path.join(HERE, "sample_data")
    os.makedirs(out, exist_ok=True)
    poses = make_poses(rng)
    poses.to_csv(os.path.join(out, "sample_poses.csv"), index=False)
    motions = make_motions(rng)
    motions.to_csv(os.path.join(out, "sample_motions.csv"), index=False)
    print(f"wrote {len(poses)} pose samples and {len(motions)} motion samples to {out}")
