"""
Saving trained models in formats other programs can use.

  ONNX           - an open standard. Works in Unreal (NNE plugin), TouchDesigner (via Python /
                   onnxruntime), Unity, C++, and many more.
  TensorFlow.js  - runs in the web browser. The Gesture Studio Live page loads this.
  forest.json    - our own tiny format for the random forest, so the browser can run it too
                   (TensorFlow.js has no random forests). Open it in a text editor: it is just
                   a list of yes/no questions!

Every export folder also gets gesture_meta.json describing the gestures and the input format.
"""
from __future__ import annotations

import json
import os
import shutil
import warnings
from datetime import datetime

import joblib
import numpy as np
import torch

from .features import FEATURE_VERSION
from .models import GestureNet, WithSoftmax


def build_meta(ds, model_type: str, accuracy: float, files: dict, run_name: str) -> dict:
    return {
        "name": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,              # "neural_net" or "random_forest"
        "mode": ds.mode,                       # "pose" or "motion"
        "classes": ds.class_names,
        "input_size": int(ds.X.shape[1]),
        "seq_len": int(ds.seq_len),
        "frame_features": int(ds.frame_features),
        "window_ms": ds.window_ms,             # motion mode: typical duration of one gesture
        "test_accuracy": round(float(accuracy), 4),
        "feature_version": FEATURE_VERSION,
        "mirror_left_hands": True,
        "files": files,
    }


def _write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- TensorFlow.js
def export_tfjs(net: GestureNet, out_dir: str, model_name="gesture_net") -> list[str]:
    """Write a TensorFlow.js 'layers model' (model.json + weights.bin) without needing TensorFlow.

    The network is a stack of Dense (fully connected) layers, which TF.js describes with a
    small JSON 'topology' plus a binary file containing every weight as float32.
    Dropout layers are left out: they only do something during training.
    """
    os.makedirs(out_dir, exist_ok=True)
    dense = net.dense_layers()
    layers, manifest, blobs = [], [], []
    for i, lin in enumerate(dense):
        name = f"dense_{i + 1}"
        last = i == len(dense) - 1
        config = {
            "name": name,
            "trainable": True,
            "dtype": "float32",
            "units": lin.out_features,
            "activation": "softmax" if last else "relu",
            "use_bias": True,
        }
        if i == 0:
            config["batch_input_shape"] = [None, lin.in_features]
        layers.append({"class_name": "Dense", "config": config})
        kernel = lin.weight.detach().cpu().numpy().T.astype("<f4")   # PyTorch stores (out, in)
        bias = lin.bias.detach().cpu().numpy().astype("<f4")
        manifest += [
            {"name": f"{name}/kernel", "shape": list(kernel.shape), "dtype": "float32"},
            {"name": f"{name}/bias", "shape": list(bias.shape), "dtype": "float32"},
        ]
        blobs += [kernel.tobytes(order="C"), bias.tobytes()]

    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    model_json = {
        "format": "layers-model",
        "generatedBy": "gesture-studio",
        "convertedBy": None,
        "modelTopology": {
            "class_name": "Sequential",
            "config": {"name": model_name, "layers": layers},
            "keras_version": "2.15.0",
            "backend": "tensorflow",
        },
        "weightsManifest": [{"paths": ["weights.bin"], "weights": manifest}],
    }
    _write_json(model_json, os.path.join(out_dir, "model.json"))
    return ["model.json", "weights.bin"]


# ---------------------------------------------------------------- ONNX
def export_onnx_net(net: GestureNet, n_inputs: int, path: str) -> bool:
    """Export the network (with softmax, so it outputs probabilities) to ONNX."""
    wrapped = WithSoftmax(net).eval()
    dummy = torch.zeros(1, n_inputs)
    kwargs = dict(input_names=["features"], output_names=["probabilities"],
                  dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}},
                  opset_version=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(wrapped, (dummy,), path, dynamo=False, **kwargs)
        except TypeError:  # older PyTorch without the `dynamo` argument
            torch.onnx.export(wrapped, (dummy,), path, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"    (ONNX export of the neural network failed: {e})")
            return False
    return True


def export_onnx_forest(rf, n_inputs: int, path: str) -> bool:
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("    (skipping random forest ONNX export: `pip install skl2onnx` to enable it)")
        return False
    onx = to_onnx(rf, initial_types=[("features", FloatTensorType([None, n_inputs]))],
                  options={id(rf): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return True


# ---------------------------------------------------------------- forest.json
def export_forest_json(rf, path: str, decimals=5) -> None:
    """Every tree as flat arrays. To predict: start at node 0; if features[feature] <= threshold
    go to `left` else `right`; stop when left == -1 and read that leaf's class probabilities."""
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        leaf_values = t.value[:, 0, :]
        leaf_values = leaf_values / np.maximum(leaf_values.sum(1, keepdims=True), 1e-12)
        is_leaf = t.children_left == -1
        trees.append({
            "feature": t.feature.tolist(),
            "threshold": np.round(t.threshold, decimals).tolist(),
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            # only leaves need probabilities; store [] for inner nodes to keep the file small
            "value": [np.round(v, 4).tolist() if leaf else [] for v, leaf in zip(leaf_values, is_leaf)],
        })
    _write_json({"format": "gesture-studio-forest", "n_classes": int(rf.n_classes_), "trees": trees}, path)


def forest_json_predict(path: str, X: np.ndarray) -> np.ndarray:
    """Reference implementation of what the browser does with forest.json (used in tests)."""
    with open(path, encoding="utf-8") as f:
        forest = json.load(f)
    out = np.zeros((len(X), forest["n_classes"]))
    for t in forest["trees"]:
        for r, x in enumerate(X):
            n = 0
            while t["left"][n] != -1:
                n = t["left"][n] if x[t["feature"][n]] <= t["threshold"][n] else t["right"][n]
            out[r] += t["value"][n]
    return out / len(forest["trees"])


# ---------------------------------------------------------------- web app index
def publish_to_web(src_dir: str, web_models_dir: str, folder_name: str, meta: dict) -> str:
    """Copy an exported model into web/public/models/ so the Live page can list it."""
    dest = os.path.join(web_models_dir, folder_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src_dir, dest)
    index_path = os.path.join(web_models_dir, "index.json")
    index = {"models": []}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    index["models"] = [m for m in index.get("models", []) if m.get("folder") != folder_name]
    index["models"].insert(0, {
        "folder": folder_name,
        "name": meta["name"],
        "mode": meta["mode"],
        "model_type": meta["model_type"],
        "classes": meta["classes"],
        "test_accuracy": meta["test_accuracy"],
        "created": meta["created"],
    })
    _write_json(index, index_path)
    return dest



# ---------------------------------------------------------------- everything at once
def export_all(ds, net, rf, ev_nn, ev_rf, winner, run_dir, name, web_dir=None, X_check=None) -> list[str]:
    """Export both models, copy the best one to <run_dir>/best (and the web app). Returns messages."""
    nn_dir, rf_dir = os.path.join(run_dir, "neural_net"), os.path.join(run_dir, "random_forest")
    for d in (nn_dir, rf_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    nn_files = {"tfjs": export_tfjs(net, nn_dir)}
    if export_onnx_net(net, ds.X.shape[1], os.path.join(nn_dir, "model.onnx")):
        nn_files["onnx"] = "model.onnx"
    nn_meta = build_meta(ds, "neural_net", ev_nn.accuracy, nn_files, name)
    _write_json(nn_meta, os.path.join(nn_dir, "gesture_meta.json"))

    export_forest_json(rf, os.path.join(rf_dir, "forest.json"))
    joblib.dump(rf, os.path.join(rf_dir, "random_forest.joblib"))
    rf_files = {"forest": "forest.json", "joblib": "random_forest.joblib"}
    if export_onnx_forest(rf, ds.X.shape[1], os.path.join(rf_dir, "model.onnx")):
        rf_files["onnx"] = "model.onnx"
    rf_meta = build_meta(ds, "random_forest", ev_rf.accuracy, rf_files, name)
    _write_json(rf_meta, os.path.join(rf_dir, "gesture_meta.json"))

    if X_check is not None:  # the browser's forest format must give the same answers as scikit-learn
        check = forest_json_predict(os.path.join(rf_dir, "forest.json"), X_check[:20])
        assert np.array_equal(check.argmax(1), rf.predict_proba(X_check[:20]).argmax(1)), "forest.json mismatch"

    best_dir, best_meta = (nn_dir, nn_meta) if winner == "neural network" else (rf_dir, rf_meta)
    best_copy = os.path.join(run_dir, "best")
    shutil.rmtree(best_copy, ignore_errors=True)
    shutil.copytree(best_dir, best_copy)

    rel = lambda p: os.path.relpath(p)  # noqa: E731
    lines = [f"Neural network -> {rel(nn_dir)}  ({', '.join(sorted(nn_files))})",
             f"Random forest  -> {rel(rf_dir)}  ({', '.join(sorted(rf_files))})",
             f"Best model ({winner}) copied to {rel(best_copy)}"]
    if web_dir:
        if os.path.isdir(web_dir):
            dest = publish_to_web(best_copy, web_dir, name, best_meta)
            lines.append(f"Also copied to {rel(dest)} - pick '{name}' on the Live page.")
        else:
            lines.append("Web app folder not found, so nothing was copied there. On the Live page, "
                         "use 'load from files' and select the files in the 'best' folder.")
    return lines
