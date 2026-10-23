"""Probabilistic forecast metrics. p = predicted P(y = 1), y in {0, 1}."""
from __future__ import annotations

import numpy as np

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
