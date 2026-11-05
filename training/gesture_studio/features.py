"""
Feature extraction: turning raw hand landmarks into numbers a model can learn from.

IMPORTANT: the web app does exactly the same maths in `web/src/lib/features.js`.
The two files must stay in sync, otherwise the model would be trained on one kind
of numbers and asked to predict on another. `tests/test_features_match_js.py`
checks that they agree.

MediaPipe gives us 21 landmarks per hand. Each landmark has
    x: 0..1 across the image width
    y: 0..1 down the image height
    z: depth relative to the wrist (roughly the same scale as x)

Landmark numbers:
    0 wrist
    1-4  thumb   (base -> tip)
    5-8  index
    9-12 middle
    13-16 ring
    17-20 pinky
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
POSE_FEATURES = NUM_LANDMARKS * 3          # 63 numbers per hand pose
MOTION_EXTRA = 3                           # dx, dy, ds per frame (see below)
FRAME_FEATURES = MOTION_EXTRA + POSE_FEATURES  # 66 numbers per frame in a motion
DEFAULT_SEQ_LEN = 16                       # frames per motion sample after resampling
FEATURE_VERSION = 1


def to_pixels(landmarks, width: float, height: float) -> np.ndarray:
    """Convert 0..1 image coordinates into pixel-like units so x and y use the same scale.

    Without this, a 640x480 camera would squash the hand horizontally.
    """
    pts = np.asarray(landmarks, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width  # MediaPipe's z uses roughly the same scale as x
    return pts


def hand_size(pts_px: np.ndarray) -> float:
    """Distance from wrist (0) to the base of the middle finger (9). Our 'ruler'."""
    return float(max(np.linalg.norm(pts_px[9] - pts_px[0]), 1e-6))


def normalize_hand(landmarks, width: float, height: float, is_left: bool,
                   mirror_left: bool = True) -> np.ndarray:
    """63 numbers describing the hand *shape*, independent of where it is and how big it looks.

    1. Move the wrist to (0, 0, 0)        -> position in the frame no longer matters
    2. Divide by the hand size            -> distance from the camera no longer matters
    3. Mirror left hands (optional)       -> left and right hands look the same to the model
    """
    pts = to_pixels(landmarks, width, height)
    rel = pts - pts[0]
    rel /= hand_size(pts)
    if is_left and mirror_left:
        rel[:, 0] *= -1.0
    return rel.reshape(-1)


def _resample(times: np.ndarray, values: np.ndarray, seq_len: int) -> np.ndarray:
    """Linearly resample a (frames, features) array to exactly `seq_len` evenly timed frames."""
    if len(times) == 1:
        return np.repeat(values, seq_len, axis=0)
    targets = np.linspace(times[0], times[-1], seq_len)
    out = np.empty((seq_len, values.shape[1]))
    j = 0
    for i, t in enumerate(targets):
        while j < len(times) - 2 and times[j + 1] < t:
            j += 1
        t0, t1 = times[j], times[j + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        a = min(max(a, 0.0), 1.0)
        out[i] = values[j] * (1 - a) + values[j + 1] * a
    return out


def motion_features(frames, seq_len: int = DEFAULT_SEQ_LEN, mirror_left: bool = True) -> np.ndarray:
    """Features for a moving gesture (e.g. a swipe).

    `frames` is a list of dicts: {"t": ms, "landmarks": 21x3, "width", "height", "is_left"}.

    For every frame we keep 66 numbers:
        dx, dy  where the wrist is compared to the first frame, measured in hand sizes.
                x is flipped so that "moving to YOUR right" is positive (like a mirror).
        ds      log(hand size / first hand size): positive = hand moving towards the camera
        63      the normalised hand shape in that frame
    Then we resample to exactly `seq_len` frames so every sample has the same length,
    and flatten to seq_len * 66 numbers.
    """
    if not frames:
        raise ValueError("motion_features needs at least one frame")
    rows, times = [], []
    ref_wrist, ref_size = None, None
    for f in frames:
        pts = to_pixels(f["landmarks"], f["width"], f["height"])
        size = hand_size(pts)
        if ref_wrist is None:
            ref_wrist, ref_size = pts[0].copy(), size
        dx = -(pts[0, 0] - ref_wrist[0]) / ref_size
        dy = (pts[0, 1] - ref_wrist[1]) / ref_size
        ds = np.log(size / ref_size)
        shape = normalize_hand(f["landmarks"], f["width"], f["height"], f["is_left"], mirror_left)
        rows.append(np.concatenate([[dx, dy, ds], shape]))
        times.append(float(f["t"]))
    return _resample(np.asarray(times), np.asarray(rows), seq_len).reshape(-1)


def pose_columns() -> list[str]:
    return [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]


def motion_columns(seq_len: int = DEFAULT_SEQ_LEN) -> list[str]:
    per_frame = ["dx", "dy", "ds"] + pose_columns()
    return [f"f{t:02d}_{c}" for t in range(seq_len) for c in per_frame]
