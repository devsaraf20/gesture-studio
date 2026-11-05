"""
Turning numbers into plain-English advice.

Every function here returns a list of sentences. `train.py` prints them and also saves them
into report.md next to the exported model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

from .data import REST_CLASS_NAMES, Dataset, Split


@dataclass
class Evaluation:
    name: str
    accuracy: float
    train_accuracy: float
    confusion: np.ndarray          # rows = true gesture, columns = predicted gesture
    recall: np.ndarray             # per class: "of all real fists, how many did we catch?"
    precision: np.ndarray          # per class: "when we said fist, how often was it a fist?"
    mean_confidence_correct: float
    mean_confidence_wrong: float


def evaluate(name, proba_test, y_test, proba_train, y_train, n_classes) -> Evaluation:
    pred = proba_test.argmax(1)
    cm = confusion_matrix(y_test, pred, labels=list(range(n_classes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.nan_to_num(np.diag(cm) / cm.sum(1))
        precision = np.nan_to_num(np.diag(cm) / cm.sum(0))
    conf = proba_test.max(1)
    right = pred == y_test
    return Evaluation(
        name=name,
        accuracy=float(right.mean()),
        train_accuracy=float((proba_train.argmax(1) == y_train).mean()),
        confusion=cm,
        recall=recall,
        precision=precision,
        mean_confidence_correct=float(conf[right].mean()) if right.any() else 0.0,
        mean_confidence_wrong=float(conf[~right].mean()) if (~right).any() else 0.0,
    )


def _pct(x):
    return f"{x * 100:.0f}%"


# ---------------------------------------------------------------- the dataset
def explain_dataset(ds: Dataset, split: Split) -> list[str]:
    counts = ds.counts()
    out = []
    kind = ("still hand poses (one snapshot per sample)" if ds.mode == "pose"
            else f"moving gestures ({ds.seq_len} frames per sample)")
    out.append(f"You have {len(ds.y)} samples of {ds.n_classes} gestures: {kind}.")
    out.append("Samples per gesture: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".")
    out.append(f"Each sample is {ds.X.shape[1]} numbers (the 'features').")
    lo, hi = min(counts.values()), max(counts.values())
    if hi > 1.5 * lo:
        small = [k for k, v in counts.items() if v == lo][0]
        big = [k for k, v in counts.items() if v == hi][0]
        out.append(f"WARNING: unbalanced data. '{big}' has {hi} samples but '{small}' only {lo}. "
                   f"The model may become biased towards '{big}'. Record more '{small}' "
                   "(or delete some of the others) so they are within about 1.5x of each other.")
    if lo < 30:
        out.append("WARNING: some gestures have fewer than 30 samples. Aim for 100+ per gesture "
                   "for a reliable model (that is only ~10 seconds of recording each).")
    if not any(n.lower() in REST_CLASS_NAMES for n in ds.class_names):
        out.append("Tip: add a 'none' gesture with random relaxed hand positions. It gives the "
                   "model a way to say 'this is nothing' instead of forcing a guess.")
    out.append(f"Split: {len(split.y_train)} for training, {len(split.y_val)} for validation, "
               f"{len(split.y_test)} for testing. {split.message}")
    return out


# ---------------------------------------------------------------- training curves
def explain_training(history: dict) -> list[str]:
    out = []
    tl, vl = history["train_loss"], history["val_loss"]
    ta, va = history["train_acc"], history["val_acc"]
    n, best = len(tl), history["best_epoch"]
    out.append(f"The neural network trained for {n} epochs (one epoch = seeing every training "
               f"sample once) in {history['seconds']:.1f} seconds.")
    out.append(f"Loss (how wrong it is, lower is better) went from {tl[0]:.2f} to {tl[-1]:.2f} on "
               f"training data and from {vl[0]:.2f} to {min(vl):.2f} at best on validation data.")
    out.append(f"The best validation loss was at epoch {best}, so we kept the weights from that epoch.")
    if best < 0.6 * n and vl[-1] > min(vl) * 1.25:
        out.append("OVERFITTING: after epoch "
                   f"{best} the training loss kept falling but the validation loss went back up. "
                   "The network started memorising the training samples instead of learning the "
                   "general shape. More varied data helps most; fewer epochs also work.")
    elif best > 0.9 * n and vl[-1] > 0.15:
        out.append("Still improving at the last epoch: try more epochs (e.g. --epochs 300).")
    if ta[-1] - va[-1] > 0.10:
        out.append(f"Training accuracy ({_pct(ta[-1])}) is well above validation accuracy "
                   f"({_pct(va[-1])}) at the end: the network is partly memorising its training samples "
                   "(overfitting). More varied data is the best cure.")
    return out


# ---------------------------------------------------------------- results
def explain_evaluation(ev: Evaluation, class_names: list[str]) -> list[str]:
    out = []
    acc = ev.accuracy
    if acc >= 0.97:
        verdict = "Excellent."
    elif acc >= 0.90:
        verdict = "Good - usable for live control, especially with smoothing."
    elif acc >= 0.75:
        verdict = "OK, but expect some wrong triggers. See the tips below."
    else:
        verdict = "Not reliable yet. The tips below say what to fix."
    out.append(f"{ev.name}: {_pct(acc)} of the test samples were recognised correctly. {verdict}")

    gap = ev.train_accuracy - acc
    if gap > 0.10:
        out.append(f"It scores {_pct(ev.train_accuracy)} on samples it learned from but only "
                   f"{_pct(acc)} on new ones: it is memorising rather than generalising (overfitting). "
                   "Record more varied samples: different angles, distances, lighting, both hands.")

    # Pairs of gestures that get mixed up
    cm = ev.confusion
    pairs = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            n_i, n_j = cm[i].sum(), cm[j].sum()
            rate = (cm[i, j] + cm[j, i]) / max(1, n_i + n_j)
            if rate >= 0.04 and (cm[i, j] + cm[j, i]) >= 1:
                pairs.append((rate, i, j))
    for rate, i, j in sorted(pairs, reverse=True)[:3]:
        a, b = class_names[i], class_names[j]
        details = [f"{cm[x, y]} '{class_names[x]}' seen as '{class_names[y]}'"
                   for x, y in ((i, j), (j, i)) if cm[x, y]]
        out.append(f"Your model confuses '{a}' and '{b}' ({', '.join(details)}). Collect more "
                   "varied samples of both - different angles, distances and both hands - and "
                   "make the difference between them clear while recording.")

    for i, r in enumerate(ev.recall):
        if cm[i].sum() and r < 0.8:
            wrong = int(np.argmax(np.where(np.arange(len(cm)) == i, -1, cm[i])))
            out.append(f"'{class_names[i]}' is only recognised {_pct(r)} of the time "
                       f"(most often mistaken for '{class_names[wrong]}').")
    for i, p in enumerate(ev.precision):
        if cm[:, i].sum() and p < 0.8:
            out.append(f"When the model says '{class_names[i]}' it is right only {_pct(p)} of the "
                       "time - it is used as a 'default guess'. Its samples may be too varied or "
                       "overlap with other gestures.")

    if ev.mean_confidence_wrong:
        out.append(f"Average confidence: {_pct(ev.mean_confidence_correct)} when right, "
                   f"{_pct(ev.mean_confidence_wrong)} when wrong. A live threshold between the two "
                   "will filter out many mistakes.")
    if not pairs and acc >= 0.97:
        out.append("No gestures are being confused with each other.")
    return out


def explain_comparison(rf: Evaluation, nn: Evaluation, winner: str) -> list[str]:
    out = [f"Random forest: {_pct(rf.accuracy)}   |   Neural network: {_pct(nn.accuracy)}"]
    if abs(rf.accuracy - nn.accuracy) < 0.01:
        out.append("Both models perform about the same. "
                   f"We picked the {winner} because it was more confident on the test set.")
    elif winner == "random forest":
        out.append("The random forest won. That is common with small datasets: forests need less "
                   "data and no tuning. As your dataset grows, the neural network often catches up.")
    else:
        out.append("The neural network won. It found smoother boundaries between the gestures "
                   "than the forest's yes/no questions could.")
    out.append("Both are exported, so you can try either in the Live page.")
    return out
