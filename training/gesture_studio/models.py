"""
The two models we compare:

1. Random forest (scikit-learn): hundreds of small decision trees that each ask yes/no
   questions like "is the index fingertip above the knuckle?" and then vote.
   Fast, needs little data, no tuning. Trains in one go - there are no "epochs".

2. Small neural network (PyTorch): layers of weighted sums + simple non-linear functions.
   It learns gradually, epoch by epoch, by nudging its weights to reduce the "loss".
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn


# ---------------------------------------------------------------- random forest
def train_random_forest(X_train, y_train, n_trees=150, seed=42) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=2,      # each final answer must be backed by at least 2 examples
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------- neural network
class GestureNet(nn.Module):
    """input -> hidden layer -> hidden layer -> one score per gesture."""

    def __init__(self, n_inputs: int, n_classes: int, hidden=(64, 32), dropout=0.2):
        super().__init__()
        self.hidden_sizes = list(hidden)
        layers, prev = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # returns raw scores ("logits")
        return self.net(x)

    def dense_layers(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]


class WithSoftmax(nn.Module):
    """Wraps the network so it outputs probabilities (0..1, summing to 1) - used for export."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.inner = net

    def forward(self, x):
        return torch.softmax(self.inner(x), dim=-1)


def default_hidden_sizes(n_inputs: int) -> tuple[int, int]:
    return (64, 32) if n_inputs <= 100 else (128, 64)


def train_neural_net(X_train, y_train, X_val, y_val, n_classes, epochs=150, lr=1e-3,
                     batch_size=32, hidden=None, seed=42, verbose=True):
    """Train the network and return (model, history).

    history has one entry per epoch for: train_loss, val_loss, train_acc, val_acc.
    We keep the weights from the epoch with the lowest validation loss ("early stopping" lite),
    so if the model starts to overfit later on, we don't keep the worse version.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden = hidden or default_hidden_sizes(X_train.shape[1])
    model = GestureNet(X_train.shape[1], n_classes, hidden=hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xtr, ytr = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    Xva, yva = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val)
    gen = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    start = time.time()

    for epoch in range(1, epochs + 1):
        # --- learn: show the network every training sample once, in small shuffled batches
        model.train()
        order = torch.randperm(len(Xtr), generator=gen)
        total_loss, correct = 0.0, 0
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            optimiser.zero_grad()
            scores = model(Xtr[b])
            loss = loss_fn(scores, ytr[b])
            loss.backward()          # work out which way to nudge each weight
            optimiser.step()         # nudge them
            total_loss += loss.item() * len(b)
            correct += (scores.argmax(1) == ytr[b]).sum().item()
        train_loss, train_acc = total_loss / len(Xtr), correct / len(Xtr)

        # --- check: how are we doing on the validation set (never used for learning)?
        model.eval()
        with torch.no_grad():
            scores = model(Xva)
            val_loss = loss_fn(scores, yva).item()
            val_acc = (scores.argmax(1) == yva).float().mean().item()

        for k, v in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[k].append(v)
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

        if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(f"    epoch {epoch:4d}/{epochs}  loss {train_loss:.3f} (val {val_loss:.3f})  "
                  f"accuracy {train_acc:6.1%} (val {val_acc:6.1%})")

    model.load_state_dict(best["state"])
    model.eval()
    history["best_epoch"] = best["epoch"]
    history["seconds"] = time.time() - start
    return model, history


def nn_predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=-1).numpy()
