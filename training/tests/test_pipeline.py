"""Quick end-to-end check of the training pipeline on the sample data.

    python tests/test_pipeline.py        (or: python -m pytest tests)
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from gesture_studio import explain, export  # noqa: E402
from gesture_studio.data import load_dataset, split_dataset  # noqa: E402
from gesture_studio.models import nn_predict_proba, train_neural_net, train_random_forest  # noqa: E402

SAMPLES = os.path.join(HERE, "..", "sample_data")


def _run(csv):
    ds = load_dataset(os.path.join(SAMPLES, csv))
    split = split_dataset(ds)
    # no sample may appear in two sets
    assert len(split.y_train) + len(split.y_val) + len(split.y_test) == len(ds.y)
    rf = train_random_forest(split.X_train, split.y_train, n_trees=20)
    net, hist = train_neural_net(split.X_train, split.y_train, split.X_val, split.y_val, ds.n_classes,
                                 epochs=15, verbose=False)
    assert len(hist["train_loss"]) == 15 and hist["train_loss"][-1] < hist["train_loss"][0]
    ev_rf = explain.evaluate("rf", rf.predict_proba(split.X_test), split.y_test,
                             rf.predict_proba(split.X_train), split.y_train, ds.n_classes)
    ev_nn = explain.evaluate("nn", nn_predict_proba(net, split.X_test), split.y_test,
                             nn_predict_proba(net, split.X_train), split.y_train, ds.n_classes)
    assert ev_rf.accuracy > 0.6 and ev_nn.accuracy > 0.6
    assert explain.explain_evaluation(ev_nn, ds.class_names)
    with tempfile.TemporaryDirectory() as tmp:
        lines = export.export_all(ds, net, rf, ev_nn, ev_rf, "neural network", tmp, "t", None, split.X_test)
        assert lines
        meta = json.load(open(os.path.join(tmp, "best", "gesture_meta.json")))
        assert meta["classes"] == ds.class_names and meta["input_size"] == ds.X.shape[1]
        model = json.load(open(os.path.join(tmp, "neural_net", "model.json")))
        n_floats = sum(int(np.prod(w["shape"])) for w in model["weightsManifest"][0]["weights"])
        assert os.path.getsize(os.path.join(tmp, "neural_net", "weights.bin")) == 4 * n_floats
        probs = export.forest_json_predict(os.path.join(tmp, "random_forest", "forest.json"), split.X_test)
        assert np.allclose(probs, rf.predict_proba(split.X_test), atol=1e-3)
    return ds


def test_pose_pipeline():
    assert _run("sample_poses.csv").mode == "pose"


def test_motion_pipeline():
    ds = _run("sample_motions.csv")
    assert ds.mode == "motion" and ds.seq_len == 16 and ds.window_ms > 0


if __name__ == "__main__":
    test_pose_pipeline()
    test_motion_pipeline()
    print("all training tests passed")
