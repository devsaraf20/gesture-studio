"""Builds gesture_studio_training.ipynb from the cells below. Run: python tools/build_notebook.py"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "gesture_studio_training.ipynb")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# ✋ Gesture Studio — training notebook

This notebook does the same as `train.py`, one step at a time, with explanations.
It works **on your laptop** (Jupyter) and in **Google Colab** (nothing to install).

**How to use it:** run each cell in order with **Shift + Enter**. Read the text between cells,
it explains what is happening. New to machine learning? Keep `LEARN.md` open next to this.

| Step | What happens |
|---|---|
| 0 | Setup (Colab downloads the project) |
| 1 | Load your gesture data |
| 2 | Split it into training, validation and test sets |
| 3 | Train a random forest |
| 4 | Train a neural network and watch it learn |
| 5 | Test both and read the plain-English verdict |
| 6 | Export the best model for the browser (TensorFlow.js) and other tools (ONNX) |
""")

md("""
## 0 · Setup

In Colab this downloads the Gesture Studio code and installs two extra packages.
On your laptop it just checks you are in the `training` folder.
""")
code("""
import os, sys

IN_COLAB = "google.colab" in sys.modules
REPO = "https://github.com/devsaraf20/gesture-studio.git"
BRANCH = "main"   # change if your code lives on another branch

if IN_COLAB:
    if not os.path.exists("/content/gesture-studio"):
        !git clone -q --depth 1 -b {BRANCH} {REPO} /content/gesture-studio
    %cd /content/gesture-studio/training
    !pip -q install onnx onnxscript skl2onnx
elif not os.path.exists("gesture_studio"):
    os.chdir(os.path.join(os.getcwd(), "training"))   # opened from the project root

sys.path.insert(0, os.getcwd())
print("Working in", os.getcwd())
""")

md("""
## 1 · Load your data

The CSV comes from the **Collect** page (*Download poses CSV* or *Download motions CSV*).
Each row is one **sample**: a `label` (the answer, e.g. `peace`) and a list of numbers
(the **features**: the normalised positions of the 21 hand points).

- Leave `USE_MY_OWN_DATA = False` to try it with the included sample data.
- Set it to `True` to upload your own CSV (Colab will show an upload button; on your laptop,
  put the CSV in the `training` folder and set `CSV_PATH`).
- For swipes and other **motions** (advanced mode) use `sample_data/sample_motions.csv`
  or your `gesture_motions.csv`. Everything below works the same.
""")
code("""
USE_MY_OWN_DATA = False
CSV_PATH = "sample_data/sample_poses.csv"     # or "sample_data/sample_motions.csv"
MODEL_NAME = None                             # name for the exported model (default: file name)

if USE_MY_OWN_DATA and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    CSV_PATH = next(iter(uploaded))

from gesture_studio.data import load_dataset, split_dataset
from gesture_studio import explain

ds = load_dataset(CSV_PATH)
MODEL_NAME = MODEL_NAME or os.path.splitext(os.path.basename(CSV_PATH))[0]
print(f"{len(ds.y)} samples, {ds.X.shape[1]} features each, mode = {ds.mode}")
print("Gestures:", ds.counts())
""")
md("Let's peek at the raw table. `x0,y0,z0` is the wrist (always 0 after normalisation), `x8,y8,z8` is the index fingertip, and so on.")
code("""
import pandas as pd
pd.read_csv(CSV_PATH).head()
""")

md("""
## 2 · Train / validation / test split

We hide some samples from the model so we can check it on data it has never seen:

- **training set**: the model learns from these
- **validation set**: checked after every epoch while training (a practice exam)
- **test set**: used once at the very end (the final exam)

If you recorded each gesture in several takes, whole takes are kept together so the test is fair.
""")
code("""
split = split_dataset(ds, seed=42)
for line in explain.explain_dataset(ds, split):
    print("-", line)
""")

md("""
## 3 · Model A: random forest

A random forest is a crowd of decision trees. Each tree asks yes/no questions about the numbers
(*"is feature y8 below -1.2?"*) and the trees vote. No epochs, no tuning: it trains in one go.
""")
code("""
from gesture_studio.models import train_random_forest
rf = train_random_forest(split.X_train, split.y_train, n_trees=150)
ev_rf = explain.evaluate("Random forest", rf.predict_proba(split.X_test), split.y_test,
                         rf.predict_proba(split.X_train), split.y_train, ds.n_classes)
print(f"Random forest test accuracy: {ev_rf.accuracy:.1%}")
""")

md("""
## 4 · Model B: a small neural network

The network turns the input numbers into one score per gesture using layers of **weights**.
Training repeats over the data in **epochs**; after each small batch it measures the **loss**
(how wrong it was) and nudges the weights to be a bit less wrong.

Try changing `EPOCHS` or `LEARNING_RATE` and re-running this cell to see what happens.
""")
code("""
from gesture_studio.models import train_neural_net, nn_predict_proba
EPOCHS = 150
LEARNING_RATE = 0.001

net, history = train_neural_net(split.X_train, split.y_train, split.X_val, split.y_val,
                                ds.n_classes, epochs=EPOCHS, lr=LEARNING_RATE)
ev_nn = explain.evaluate("Neural network", nn_predict_proba(net, split.X_test), split.y_test,
                         nn_predict_proba(net, split.X_train), split.y_train, ds.n_classes)
""")
md("""
### Training curves

- **Loss** should go down. If the *validation* line turns back up while *training* keeps going down,
  the model is **overfitting** (memorising instead of learning).
- **Accuracy** is the share of samples it gets right.
- The dashed line marks the epoch whose weights we kept (lowest validation loss).
""")
code("""
%matplotlib inline
from gesture_studio import plots
os.makedirs(f"exports/{MODEL_NAME}", exist_ok=True)
plots.plot_history(history, f"exports/{MODEL_NAME}/training_curves.png");
for line in explain.explain_training(history):
    print("-", line)
""")

md("""
## 5 · Test both models

A **confusion matrix** shows real gestures (rows) against predictions (columns).
Numbers on the diagonal are correct answers; anything else is a mix-up.
""")
code("""
plots.plot_confusions([ev_rf, ev_nn], ds.class_names, f"exports/{MODEL_NAME}/confusion_matrices.png");
""")
code("""
for ev in (ev_rf, ev_nn):
    for line in explain.explain_evaluation(ev, ds.class_names):
        print("-", line)
    print()

if abs(ev_rf.accuracy - ev_nn.accuracy) < 0.005:
    winner = "neural network" if ev_nn.mean_confidence_correct >= ev_rf.mean_confidence_correct else "random forest"
else:
    winner = "neural network" if ev_nn.accuracy > ev_rf.accuracy else "random forest"
for line in explain.explain_comparison(ev_rf, ev_nn, winner):
    print("-", line)
""")

md("""
## 6 · Export

Both models are saved, and the best one is copied to `exports/<name>/best/`:

- `model.json` + `weights.bin`: TensorFlow.js, for the Gesture Studio **Live** page (neural network)
- `forest.json`: the random forest in a format the Live page can run
- `model.onnx`: ONNX, for Unreal (NNE), TouchDesigner, Unity and others
- `gesture_meta.json`: the gesture names and input format

On your laptop the best model is also copied into `web/public/models/`, so it appears in the Live page's list.
In Colab, the last cell downloads a zip: unzip it, then on the Live page click **load from files**
and select every file inside the `best` folder.
""")
code("""
from gesture_studio import export
web_dir = os.path.abspath(os.path.join("..", "web", "public", "models"))
for line in export.export_all(ds, net, rf, ev_nn, ev_rf, winner, f"exports/{MODEL_NAME}", MODEL_NAME,
                              web_dir if not IN_COLAB else None, split.X_test):
    print("-", line)
""")
code("""
if IN_COLAB:
    import shutil
    from google.colab import files
    shutil.make_archive(f"/content/{MODEL_NAME}", "zip", f"exports/{MODEL_NAME}")
    files.download(f"/content/{MODEL_NAME}.zip")
""")

md("""
## What next?

- **Accuracy low or gestures confused?** Record more varied samples of the confused gestures
  (angles, distances, both hands, different lighting), export again and re-run.
- **Overfitting?** More varied data helps most. Fewer epochs also helps.
- **Swipes?** Set `CSV_PATH` to a motions CSV in step 1 and run everything again.
- Read `LEARN.md` for the ideas behind every step.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
nbf.write(nb, OUT)
print("wrote", os.path.normpath(OUT))
