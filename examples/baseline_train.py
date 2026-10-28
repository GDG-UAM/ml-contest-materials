"""Train the baseline model on an exported training file.

    python examples/baseline_train.py [--data data/train_100k.npz] [--out examples/baseline_model.pkl]

1. Compute the features of every training sample, exactly as ``Model.predict`` will.
2. Hold out a random validation split and report the accuracy on it.
3. Refit on all the data and save the model next to ``baseline_model.py``.

The validation split comes from the training distribution. The evaluation data may differ,
so expect the official score to be lower.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Paths are built from this file's location, so the script works from any working directory.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # makes competition_api importable

from baseline_model import MODEL_FILE, features  # noqa: E402
from competition_api import Dataset, load_dataset  # noqa: E402

DEFAULT_DATA = REPO / "data" / "train_100k.npz"


def feature_matrix(ds: Dataset) -> np.ndarray:
    """Stack the feature vectors of every sample in ``ds`` into an (N, F) matrix."""
    start = time.perf_counter()
    X = np.stack([features(sample) for sample in ds])
    print(f"features: {X.shape[1]} per sample for {len(ds)} samples ({time.perf_counter() - start:.0f} s)")
    return X


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="exported training file (.npz or .parquet)")
    parser.add_argument("--out", type=Path, default=MODEL_FILE, help="where to save the trained model")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of rows held out (0 skips validation)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the split and of the classifier")
    args = parser.parse_args(argv)

    ds = load_dataset(args.data)
    X, y = feature_matrix(ds), np.asarray(ds.labels)

    # Hold out a random subset of the rows for validation.
    order = np.random.default_rng(args.seed).permutation(len(y))
    n_val = int(args.val_fraction * len(y))
    val, train = order[:n_val], order[n_val:]
    if n_val:
        classifier = make_classifier(args.seed).fit(X[train], y[train])
        accuracy = np.mean(classifier.predict(X[val]) == y[val])
        print(f"validation accuracy: {accuracy:.3f} on {n_val} samples")

    classifier = make_classifier(args.seed).fit(X, y)  # final model: all the data
    with open(args.out, "wb") as fh:
        pickle.dump(classifier, fh)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
