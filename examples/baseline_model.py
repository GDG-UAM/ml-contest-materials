"""Baseline model: generic time-series features + gradient boosting (scikit-learn).

Each sample is summarized by a fixed-length feature vector:

* level and scale: mean and standard deviation of each channel,
* distribution: quantiles of each standardized channel,
* roughness: size of the step-to-step changes,
* memory: autocorrelation at a few lags,
* spectrum: power in equal-width frequency bands,
* cross-channel: correlation between every pair of channels,
* metadata: the numeric fields of ``x.meta``,
* missing data: fraction of NaN values.

A ``HistGradientBoostingClassifier`` trained on these vectors makes the predictions.

    python examples/baseline_train.py                                # writes examples/baseline_model.pkl
    python harness.py evaluate --model examples/baseline_model.py    # evaluates it locally
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from competition_api import Sample

# The trained model lives next to this file and is found relative to it, so loading works
# whatever the working directory is (locally and in the official evaluation).
MODEL_FILE = Path(__file__).with_name("baseline_model.pkl")

QUANTILES = (5, 25, 50, 75, 95)   # percentiles of each standardized channel
ACF_LAGS = (1, 2, 4, 8, 16, 32)   # autocorrelation lags, in time steps
N_BANDS = 8                       # equal-width bands the power spectrum is split into


# ---------------------------------------------------------------------- preprocessing


def fill_missing(series: np.ndarray) -> np.ndarray:
    """Return a float64 copy of ``series`` with each channel's NaNs linearly interpolated.

    Gaps at the edges take the nearest valid value; a channel with no valid value becomes 0.
    """
    filled = np.array(series, dtype=np.float64)
    t = np.arange(filled.shape[1])
    for channel in filled:  # rows are views, so this edits ``filled`` in place
        missing = np.isnan(channel)
        if missing.all():
            channel[:] = 0.0
        elif missing.any():
            channel[missing] = np.interp(t[missing], t[~missing], channel[~missing])
    return filled


def standardize(series: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance per channel (a constant channel becomes all zeros)."""
    centered = series - series.mean(axis=1, keepdims=True)
    return centered / (series.std(axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------- feature groups
# Each per-channel group maps a (C, L) array to a (C, n_features) array. All but the first
# work on the standardized signal, where autocorrelation and spectra are comparable.


def level_and_scale_features(raw: np.ndarray) -> np.ndarray:
    return np.stack([raw.mean(axis=1), raw.std(axis=1)], axis=1)


def distribution_features(z: np.ndarray) -> np.ndarray:
    return np.percentile(z, QUANTILES, axis=1).T


def roughness_features(z: np.ndarray) -> np.ndarray:
    steps = np.diff(z, axis=1)
    return np.stack([steps.std(axis=1), np.abs(steps).mean(axis=1)], axis=1)


def autocorrelation_features(z: np.ndarray) -> np.ndarray:
    return np.stack([np.mean(z[:, lag:] * z[:, :-lag], axis=1) for lag in ACF_LAGS], axis=1)


def spectral_features(z: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(z, axis=1))[:, 1:] ** 2  # skip the zero-frequency term
    bands = np.array_split(power, N_BANDS, axis=1)
    return np.log1p(np.stack([band.sum(axis=1) for band in bands], axis=1))


def cross_channel_features(z: np.ndarray) -> np.ndarray:
    """Correlation of every pair of channels: C * (C - 1) / 2 values."""
    correlation = z @ z.T / z.shape[1]
    return correlation[np.triu_indices(len(z), k=1)]


def meta_features(meta: dict) -> list[float]:
    """Numeric metadata fields in a fixed (alphabetical) order. Text fields are left out."""
    return [float(value) for _, value in sorted(meta.items()) if isinstance(value, (int, float))]


SHAPE_GROUPS = (distribution_features, roughness_features, autocorrelation_features, spectral_features)


def features(x: Sample) -> np.ndarray:
    """Feature vector of one sample. Used for training and for prediction alike."""
    raw = fill_missing(x.series)
    z = standardize(raw)
    per_channel = np.hstack([level_and_scale_features(raw), *(group(z) for group in SHAPE_GROUPS)])
    vector = np.concatenate([
        per_channel.ravel(),
        cross_channel_features(z),
        meta_features(x.meta),
        [np.isnan(x.series).mean()],
    ])
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------- model


class Model:
    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"{MODEL_FILE} not found: run examples/baseline_train.py first")
        with open(MODEL_FILE, "rb") as fh:
            self.classifier = pickle.load(fh)

    def predict(self, x: Sample) -> int:
        return int(self.classifier.predict(features(x)[np.newaxis, :])[0])
