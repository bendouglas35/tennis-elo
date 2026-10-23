"""Probabilistic forecast metrics. p = predicted P(y = 1), y in {0, 1}."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def accuracy(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


# ---- per-match losses and cluster bootstrap -------------------------------------------------

def log_loss_vec(y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_vec(y, p) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def accuracy_vec(y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    hit = (p > 0.5) == (y > 0.5)
    return np.where(p == 0.5, 0.5, hit.astype(float))


def cluster_bootstrap(groups, values: dict, diffs=(), n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the mean of each per-match array, resampling whole groups (tournaments).

    values: name -> per-match array. diffs: (a, b) pairs; the paired difference mean(a) - mean(b)
    is bootstrapped on the same draws. Returns name -> (mean, lo, hi) with a 95% percentile interval;
    difference keys are "a - b".
    """
    codes = pd.factorize(np.asarray(groups))[0]
    g = codes.max() + 1
    counts = np.bincount(codes, minlength=g).astype(float)
    names = list(values)
    sums = np.column_stack([np.bincount(codes, weights=np.asarray(values[k], float), minlength=g)
                            for k in names])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(n_boot, g))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]      # (n_boot, n_names)
    out = {}
    for j, k in enumerate(names):
        out[k] = (float(sums[:, j].sum() / counts.sum()), *np.percentile(boot[:, j], [2.5, 97.5]))
    pos = {k: j for j, k in enumerate(names)}
    for a, b in diffs:
        d = boot[:, pos[a]] - boot[:, pos[b]]
        point = float(sums[:, pos[a]].sum() / counts.sum() - sums[:, pos[b]].sum() / counts.sum())
        out[f"{a} - {b}"] = (point, *np.percentile(d, [2.5, 97.5]))
    return out
