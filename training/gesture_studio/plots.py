"""Charts: training curves and confusion matrices (saved as PNG, also shown in the notebook)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TRAIN_COLOUR = "#2a78d6"  # blue
VAL_COLOUR = "#eb6834"    # orange
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#a9c8ef", "#2a78d6", "#123f78"])


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_history(history: dict, path: str | None = None):
    """Two charts side by side: loss (lower is better) and accuracy (higher is better)."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best = history["best_epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    panels = [
        (ax1, "train_loss", "val_loss", "Loss - how wrong the model is (lower = better)", "loss"),
        (ax2, "train_acc", "val_acc", "Accuracy - share of samples right (higher = better)", "accuracy"),
    ]
    for ax, tk, vk, title, ylabel in panels:
        _style(ax, title, ylabel)
        ax.plot(epochs, history[tk], color=TRAIN_COLOUR, linewidth=2)
        ax.plot(epochs, history[vk], color=VAL_COLOUR, linewidth=2)
        # direct labels at the line ends instead of a legend box
        ax.annotate("training", (epochs[-1], history[tk][-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=9)
        ax.annotate("validation", (epochs[-1], history[vk][-1]), xytext=(6, -10 if ax is ax1 else 10),
                    textcoords="offset points", color=INK, va="center", fontsize=9)
        ax.axvline(best, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"kept weights\nfrom epoch {best}", (best, ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=8, color=INK_2)
        ax.set_xlim(1, epochs[-1] * 1.12)
    ax2.set_ylim(0, 1.02)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    return fig


def plot_confusions(evaluations, class_names, path: str | None = None):
    """One confusion matrix per model. Rows = the real gesture, columns = what the model said.

    A perfect model has dark squares only on the diagonal.
    """
    n = len(evaluations)
    size = max(4.2, 0.65 * len(class_names) + 2.4)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, evaluations):
        cm = ev.confusion
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.set_title(f"{ev.name}  ({ev.accuracy:.0%} correct)", loc="left", color=INK,
                     fontsize=12, fontweight="bold")
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks, class_names, rotation=35, ha="right", color=INK_2)
        ax.set_yticks(ticks, class_names, color=INK_2)
        ax.set_xlabel("model predicted", color=INK_2)
        ax.set_ylabel("real gesture", color=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if cm[i, j] == 0:
                    continue
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if share[i, j] > 0.55 else INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    return fig
