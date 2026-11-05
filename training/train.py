"""
Gesture Studio - training script.

    python train.py                                   # trains on the included sample data
    python train.py path/to/gesture_poses.csv         # your own pose dataset
    python train.py path/to/gesture_motions.csv       # your own motion (swipe) dataset
    python train.py my.csv --epochs 300 --name my_poses_v2

What it does, step by step:
  1. loads the CSV you exported from the Collect page
  2. splits it into training / validation / test samples
  3. trains a random forest
  4. trains a small neural network, epoch by epoch
  5. tests both on samples they have never seen, and explains the results in plain English
  6. exports both models (ONNX + TensorFlow.js + forest.json) and copies the best one
     into web/public/models/ so the Live page can use it straight away
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gesture_studio import explain, export  # noqa: E402
from gesture_studio.data import load_dataset, split_dataset  # noqa: E402
from gesture_studio.models import nn_predict_proba, train_neural_net, train_random_forest  # noqa: E402


def say(lines, indent="  "):
    for line in lines:
        print(textwrap.fill(line, width=92, initial_indent=indent + "- ", subsequent_indent=indent + "  "))


def step(n, title):
    print(f"\n{'=' * 92}\nSTEP {n}/6  {title}\n{'=' * 92}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train gesture recognition models.")
    p.add_argument("csv", nargs="?", default=os.path.join(HERE, "sample_data", "sample_poses.csv"),
                   help="dataset CSV exported from the Gesture Studio Collect page")
    p.add_argument("--name", help="name for this model (default: the CSV file name)")
    p.add_argument("--epochs", type=int, default=150, help="neural network training rounds")
    p.add_argument("--lr", type=float, default=1e-3, help="neural network learning rate")
    p.add_argument("--trees", type=int, default=150, help="number of trees in the random forest")
    p.add_argument("--out", default=os.path.join(HERE, "exports"), help="where to save results")
    p.add_argument("--web-dir", default=os.path.join(HERE, "..", "web", "public", "models"),
                   help="copy the best model here for the web app ('' to skip)")
    p.add_argument("--seed", type=int, default=42, help="random seed (same seed = same result)")
    p.add_argument("--show", action="store_true", help="open the charts in windows as well")
    args = p.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")
    from gesture_studio import plots  # imported after choosing the matplotlib backend

    if not os.path.exists(args.csv):
        sys.exit(f"Could not find '{args.csv}'. Export a CSV from the Collect page first.")
    name = args.name or os.path.splitext(os.path.basename(args.csv))[0]
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)
    report = [f"# Gesture Studio training report: {name}\n", f"Dataset: `{os.path.basename(args.csv)}`\n"]

    def section(title, lines):
        say(lines)
        report.append(f"\n## {title}\n")
        report.extend(f"- {line}" for line in lines)

    # ------------------------------------------------------------------ 1. data
    step(1, "Loading your data")
    ds = load_dataset(args.csv)
    split = split_dataset(ds, seed=args.seed)
    section("Your data", explain.explain_dataset(ds, split))
    if ds.n_classes < 2:
        sys.exit("You need at least 2 gestures to train a classifier.")

    # ------------------------------------------------------------------ 2. split
    step(2, "Splitting into training / validation / test sets")
    say([f"training:   {len(split.y_train):5d} samples - the model learns from these",
         f"validation: {len(split.y_val):5d} samples - checked after every epoch, never learned from",
         f"test:       {len(split.y_test):5d} samples - the final exam, used once at the end"])

    # ------------------------------------------------------------------ 3. random forest
    step(3, f"Training a random forest ({args.trees} decision trees)")
    rf = train_random_forest(split.X_train, split.y_train, n_trees=args.trees, seed=args.seed)
    ev_rf = explain.evaluate("Random forest", rf.predict_proba(split.X_test), split.y_test,
                             rf.predict_proba(split.X_train), split.y_train, ds.n_classes)
    say([f"done - {ev_rf.accuracy:.1%} correct on the test set"])

    # ------------------------------------------------------------------ 4. neural network
    step(4, f"Training a neural network ({args.epochs} epochs)")
    net, history = train_neural_net(split.X_train, split.y_train, split.X_val, split.y_val,
                                    ds.n_classes, epochs=args.epochs, lr=args.lr, seed=args.seed)
    ev_nn = explain.evaluate("Neural network", nn_predict_proba(net, split.X_test), split.y_test,
                             nn_predict_proba(net, split.X_train), split.y_train, ds.n_classes)
    plots.plot_history(history, os.path.join(run_dir, "training_curves.png"))
    section("How the neural network trained", explain.explain_training(history))
    report.append("\n![training curves](training_curves.png)\n")

    # ------------------------------------------------------------------ 5. compare
    step(5, "Testing both models on samples they have never seen")
    plots.plot_confusions([ev_rf, ev_nn], ds.class_names, os.path.join(run_dir, "confusion_matrices.png"))
    report.append("\n![confusion matrices](confusion_matrices.png)\n")
    section("Random forest results", explain.explain_evaluation(ev_rf, ds.class_names))
    print()
    section("Neural network results", explain.explain_evaluation(ev_nn, ds.class_names))
    if abs(ev_rf.accuracy - ev_nn.accuracy) < 0.005:
        winner = "neural network" if ev_nn.mean_confidence_correct >= ev_rf.mean_confidence_correct else "random forest"
    else:
        winner = "neural network" if ev_nn.accuracy > ev_rf.accuracy else "random forest"
    print()
    section("Which model won?", explain.explain_comparison(ev_rf, ev_nn, winner))

    # ------------------------------------------------------------------ 6. export
    step(6, "Exporting the models")
    web_dir = args.web_dir and os.path.abspath(args.web_dir)
    lines = export.export_all(ds, net, rf, ev_nn, ev_rf, winner, run_dir, name, web_dir, split.X_test)
    section("Exported files", lines)

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nAll done. Charts and a written report are in {os.path.relpath(run_dir)}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return {"rf": ev_rf, "nn": ev_nn, "winner": winner, "run_dir": run_dir}


if __name__ == "__main__":
    main()
