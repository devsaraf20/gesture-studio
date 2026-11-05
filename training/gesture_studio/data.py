"""
Loading the CSV exported by the web app and splitting it into train / validation / test sets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FRAME_FEATURES, POSE_FEATURES

META_COLUMNS = {"label", "take", "duration_ms"}
REST_CLASS_NAMES = {"none", "idle", "nothing", "background", "rest", "neutral"}


@dataclass
class Dataset:
    X: np.ndarray                 # (samples, features) - the inputs
    y: np.ndarray                 # (samples,) - the answers, as class numbers 0..C-1
    class_names: list[str]        # class number -> name
    takes: np.ndarray             # which recording "take" each sample came from
    mode: str                     # "pose" (still hand) or "motion" (moving hand)
    seq_len: int = 1              # frames per sample (motion mode)
    frame_features: int = POSE_FEATURES
    window_ms: float | None = None  # typical length of a motion, used for live use
    feature_columns: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def counts(self) -> dict[str, int]:
        return {name: int((self.y == i).sum()) for i, name in enumerate(self.class_names)}


def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} has no 'label' column. Was it exported from Gesture Studio?")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str)

    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    is_motion = any(re.match(r"^f\d+_", c) for c in feature_cols)
    if is_motion:
        seq_len = len({c.split("_")[0] for c in feature_cols})
        frame_features = len(feature_cols) // seq_len
        mode = "motion"
    else:
        seq_len, frame_features, mode = 1, len(feature_cols), "pose"
    if mode == "pose" and frame_features != POSE_FEATURES:
        print(f"  note: expected {POSE_FEATURES} pose columns, found {frame_features}")
    if mode == "motion" and frame_features != FRAME_FEATURES:
        print(f"  note: expected {FRAME_FEATURES} values per frame, found {frame_features}")

    bad = df[feature_cols].isna().any(axis=1)
    if bad.any():
        print(f"  note: skipped {int(bad.sum())} rows with missing numbers")
        df = df[~bad]

    class_names = sorted(df["label"].unique().tolist())
    index = {n: i for i, n in enumerate(class_names)}
    takes = df["take"].astype(str).to_numpy() if "take" in df.columns else np.arange(len(df)).astype(str)
    window_ms = float(df["duration_ms"].median()) if "duration_ms" in df.columns else None

    return Dataset(
        X=df[feature_cols].to_numpy(dtype=np.float32),
        y=df["label"].map(index).to_numpy(dtype=np.int64),
        class_names=class_names,
        takes=takes,
        mode=mode,
        seq_len=seq_len,
        frame_features=frame_features,
        window_ms=window_ms,
        feature_columns=feature_cols,
    )


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    by_take: bool          # True if whole recording takes were kept together (a fairer test)
    message: str


def split_dataset(ds: Dataset, val_fraction=0.15, test_fraction=0.15, seed=42) -> Split:
    """Split into train (to learn from), validation (to watch during training) and test (final exam).

    Samples recorded in one press of the record button ("a take") are almost identical to
    each other. If some of them end up in training and some in testing, the test becomes
    too easy - like seeing the exam questions in advance. So when there are enough takes,
    we keep each take entirely in one set.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    takes_per_class = {c: np.unique(ds.takes[ds.y == c]) for c in range(ds.n_classes)}
    by_take = all(len(t) >= 3 for t in takes_per_class.values())

    for c in range(ds.n_classes):
        idx_c = np.where(ds.y == c)[0]
        if by_take:
            takes = takes_per_class[c].copy()
            rng.shuffle(takes)
            groups = [idx_c[ds.takes[idx_c] == t] for t in takes]
            total = len(idx_c)
            buckets = {"test": [], "val": [], "train": []}
            filled = {"test": 0, "val": 0}
            for g in groups:
                # Fill test first, then validation, then everything else goes to training.
                # (Each set gets at least one take.)
                if not buckets["test"] or filled["test"] + len(g) / 2 <= test_fraction * total:
                    buckets["test"].append(g)
                    filled["test"] += len(g)
                elif not buckets["val"] or filled["val"] + len(g) / 2 <= val_fraction * total:
                    buckets["val"].append(g)
                    filled["val"] += len(g)
                else:
                    buckets["train"].append(g)
            if not buckets["train"]:  # always keep something to learn from
                buckets["train"].append(buckets["val"].pop())
            test_idx += list(np.concatenate(buckets["test"]))
            val_idx += list(np.concatenate(buckets["val"])) if buckets["val"] else []
            train_idx += list(np.concatenate(buckets["train"]))
        else:
            idx_c = rng.permutation(idx_c)
            n_test = max(1, int(round(test_fraction * len(idx_c))))
            n_val = max(1, int(round(val_fraction * len(idx_c))))
            if len(idx_c) - n_test - n_val < 1:
                raise ValueError(f"Class '{ds.class_names[c]}' has only {len(idx_c)} samples - record at least 10.")
            test_idx += list(idx_c[:n_test])
            val_idx += list(idx_c[n_test:n_test + n_val])
            train_idx += list(idx_c[n_test + n_val:])

    if by_take:
        message = ("Each recording take was kept entirely in one set, so the test score is a fair "
                   "estimate of how well the model handles new recordings.")
    else:
        few = [ds.class_names[c] for c, t in takes_per_class.items() if len(t) < 3]
        message = ("Samples were split randomly because some classes were recorded in fewer than 3 takes "
                   f"({', '.join(few)}). Neighbouring samples from one take look almost identical, so the "
                   "test score may be too optimistic. Tip: record each gesture in several short takes.")

    def pick(ix):
        ix = np.asarray(ix, dtype=int)
        return ds.X[ix], ds.y[ix]

    Xtr, ytr = pick(train_idx)
    Xva, yva = pick(val_idx)
    Xte, yte = pick(test_idx)
    return Split(Xtr, ytr, Xva, yva, Xte, yte, by_take, message)
