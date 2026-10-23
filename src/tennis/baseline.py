"""Baseline A: the higher-ranked player wins, plus a probability version via 1-D logistic regression.

Only train (<= TRAIN_END_YEAR) and validation (VAL_YEAR) years are touched here. The frozen test
years (>= TEST_START_YEAR) are deliberately not evaluated by this module.
"""
from __future__ import annotations

import duckdb
import numpy as np

from . import TRAIN_END_YEAR, VAL_YEAR
from .metrics import accuracy, brier, log_loss


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-6, iters: int = 50) -> np.ndarray:
    """Newton-Raphson logistic regression with intercept. Returns [intercept, slope]."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = _sigmoid(X @ beta)
        w = p * (1 - p)
        grad = X.T @ (y - p) - l2 * beta
        hess = X.T @ (X * w[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(beta[0] + beta[1] * x)


def run(db_path, include_team_events: bool = False) -> dict:
    """Team events (Davis/Laver/ATP/United Cup) and Next Gen Finals are excluded by default."""
    population = "" if include_team_events else "and not is_team_event and not is_nonstandard_format"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            f"""
            select year(tourney_date) as yr, rank_a, rank_b, a_wins
            from match_sides
            where rank_a is not null and rank_b is not null and rank_a <> rank_b
              and year(tourney_date) <= ?
              {population}
            """,
            [VAL_YEAR],
        ).df()
    finally:
        con.close()

    df["x"] = np.log(df["rank_b"] / df["rank_a"])  # > 0 when A is ranked higher (smaller number)
    y = df["a_wins"].astype(float).to_numpy()
    train, val = (df["yr"] <= TRAIN_END_YEAR).to_numpy(), (df["yr"] == VAL_YEAR).to_numpy()
    if not train.any() or not val.any():
        raise SystemExit("Need data in both the training years and the validation year.")

    beta = fit_logistic(df.loc[train, "x"].to_numpy(), y[train])
    p_val = predict(beta, df.loc[val, "x"].to_numpy())
    y_val = y[val]
    higher_ranked_wins = (df.loc[val, "x"].to_numpy() > 0).astype(float)
    return {
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "beta": beta.tolist(),
        "val_accuracy_higher_ranked": accuracy(y_val, higher_ranked_wins),
        "val_log_loss": log_loss(y_val, p_val),
        "val_brier": brier(y_val, p_val),
        "val_accuracy_logistic": accuracy(y_val, p_val),
    }
