"""v0.4 models: antisymmetric logistic regression and gradient boosting.

Both models are exactly antisymmetric under the player mirror, p(x) + p(mirror(x)) = 1, so the A/B
labelling cannot matter:
* logistic regression uses only ANTI columns, no intercept, scaling without centring;
* LightGBM is trained on the data plus its mirror image and predicts the average of f(x) and
  1 - f(mirror(x)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .dataset import ANTI, mirror
from .metrics import log_loss

CLIP = 5.0


class LRModel:
    """L2 logistic regression on antisymmetric columns (symmetric columns cannot matter here)."""

    def __init__(self, columns, C: float = 0.1):
        self.columns = [c for c in columns if c in set(ANTI)]
        self.C = C

    def fit(self, tr: pd.DataFrame, y=None):
        X = tr[self.columns]
        self.scale_ = X.std(skipna=True).replace(0, 1.0).fillna(1.0)
        Z = self._z(X)
        y = tr["y"].to_numpy() if y is None else y
        # mirrored copy makes the fit exactly symmetric even when the solver stops early
        self.m_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000)
        self.m_.fit(np.vstack([Z, -Z]), np.concatenate([y, 1 - y]))
        return self

    def _z(self, X):
        return (X / self.scale_).clip(-CLIP, CLIP).fillna(0.0).to_numpy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.m_.predict_proba(self._z(df[self.columns]))[:, 1]

    def coef(self) -> pd.Series:
        return pd.Series(self.m_.coef_[0], index=self.columns)


GBM_DEFAULT = dict(num_leaves=8, min_child_samples=100, learning_rate=0.03, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0)


class GBMModel:
    def __init__(self, columns, n_estimators: int = 400, **params):
        self.columns = list(columns)
        self.n_estimators = n_estimators
        self.params = {**GBM_DEFAULT, **params}

    def fit(self, tr: pd.DataFrame):
        import lightgbm as lgb

        X = tr[self.columns]
        y = tr["y"].to_numpy()
        self.m_ = lgb.LGBMClassifier(n_estimators=self.n_estimators, objective="binary", n_jobs=2,
                                     verbose=-1, random_state=0, **self.params)
        self.m_.fit(pd.concat([X, mirror(X)], ignore_index=True), np.concatenate([y, 1 - y]))
        return self

    def predict(self, df: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        X = df[self.columns]
        f = self.m_.predict_proba(X, num_iteration=num_iteration)[:, 1]
        g = self.m_.predict_proba(mirror(X), num_iteration=num_iteration)[:, 1]
        return 0.5 * (f + 1.0 - g)

    def importances(self) -> pd.Series:
        return pd.Series(self.m_.booster_.feature_importance("gain"), index=self.columns)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    """Slope-only recalibration on the logit scale (no intercept keeps antisymmetry)."""

    def fit(self, p, y):
        z = logit(np.asarray(p))[:, None]
        self.m_ = LogisticRegression(C=1e6, fit_intercept=False).fit(np.vstack([z, -z]),
                                                                    np.concatenate([y, 1 - y]))
        self.slope_ = float(self.m_.coef_[0][0])
        return self

    def predict(self, p) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.slope_ * logit(np.asarray(p))))


def rolling_origin(df: pd.DataFrame, make, years, predict_kwargs=None) -> pd.DataFrame:
    """Expanding-window evaluation: for each year, fit on all earlier years and predict that year.

    Returns match_id, yr, tourney_id, y, p for the evaluated years.
    """
    out = []
    for yr in years:
        tr, ev = df[df["yr"] < yr], df[df["yr"] == yr]
        m = make().fit(tr)
        p = m.predict(ev, **(predict_kwargs or {}))
        out.append(pd.DataFrame({"match_id": ev["match_id"].to_numpy(), "yr": yr,
                                 "tourney_id": ev["tourney_id"].to_numpy(),
                                 "y": ev["y"].to_numpy(), "p": p}))
    return pd.concat(out, ignore_index=True)


def tune_lr(df, columns, years, grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)) -> pd.DataFrame:
    rows = []
    for C in grid:
        r = rolling_origin(df, lambda: LRModel(columns, C), years)
        rows.append({"C": C, "log_loss": log_loss(r["y"], r["p"])})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def tune_gbm(df, columns, years, configs, max_iter=800, step=50, verbose=True) -> pd.DataFrame:
    """Rolling-origin tuning. For every config the number of boosting rounds is chosen on the same
    folds by scoring predictions at every `step` rounds."""
    iters = list(range(step, max_iter + 1, step))
    rows = []
    for ci, cfg in enumerate(configs):
        preds = {k: [] for k in iters}
        ys = []
        for yr in years:
            tr, ev = df[df["yr"] < yr], df[df["yr"] == yr]
            m = GBMModel(columns, n_estimators=max_iter, **cfg).fit(tr)
            for k in iters:
                preds[k].append(m.predict(ev, num_iteration=k))
            ys.append(ev["y"].to_numpy())
        y = np.concatenate(ys)
        for k in iters:
            rows.append({"cfg": ci, **cfg, "n_estimators": k,
                         "log_loss": log_loss(y, np.concatenate(preds[k]))})
        if verbose:
            best = min(r["log_loss"] for r in rows if r["cfg"] == ci)
            print(f"  gbm config {ci} {cfg}: best rolling log loss {best:.5f}", flush=True)
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
