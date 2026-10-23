"""v0.4 experiments. Development functions only ever load years <= VAL_YEAR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import TRAIN_END_YEAR, VAL_YEAR
from . import dataset, models
from .metrics import log_loss

TUNE_YEARS = [2018, 2019, 2020, 2021]
CONFIG_PATH = Path("docs/model_config.json")
# First pass (800 rounds max) preferred 16 leaves and the full 800 rounds, i.e. the edge of the grid,
# so the final pass widens it: more rounds and bigger trees.
GBM_CONFIGS = [
    dict(num_leaves=16, min_child_samples=200),
    dict(num_leaves=32, min_child_samples=200),
    dict(num_leaves=16, min_child_samples=100),
]
MAX_ITER = 1600


def dev_frame(db_path) -> pd.DataFrame:
    df = dataset.load(db_path, VAL_YEAR)
    dataset.assert_no_test_years(df)
    return df


def tune(db_path, path=CONFIG_PATH, verbose=True) -> dict:
    """Choose hyperparameters by expanding-window evaluation on 2018-2021 (train < each year)."""
    df = dev_frame(db_path)
    cols = dataset.columns_for(dataset.FULL_GROUPS)
    lr = models.tune_lr(df, cols, TUNE_YEARS)
    if verbose:
        print("LR tuning:\n", lr.to_string(index=False), flush=True)
    best_C = float(lr.iloc[0]["C"])
    gb = models.tune_gbm(df, cols, TUNE_YEARS, GBM_CONFIGS, max_iter=MAX_ITER, step=100, verbose=verbose)
    top = gb.iloc[0]
    cfg = {k: (int(top[k]) if k in ("num_leaves", "min_child_samples") else float(top[k]))
           for k in ("num_leaves", "min_child_samples", "colsample_bytree", "reg_lambda")
           if pd.notna(top.get(k))}
    cfg = {**{k: v for k, v in models.GBM_DEFAULT.items() if k not in cfg}, **cfg}
    n_est = int(top["n_estimators"])
    # out-of-fold GBM predictions on the tuning years -> slope-only recalibration
    oof = models.rolling_origin(df, lambda: models.GBMModel(cols, n_est, **cfg), TUNE_YEARS)
    platt = models.Platt().fit(oof["p"].to_numpy(), oof["y"].to_numpy())
    out = {"lr_C": best_C, "lr_rolling_log_loss": float(lr.iloc[0]["log_loss"]),
           "gbm_params": cfg, "gbm_n_estimators": n_est,
           "gbm_rolling_log_loss": float(top["log_loss"]), "gbm_platt_slope": platt.slope_,
           "gbm_oof_log_loss_raw": log_loss(oof["y"], oof["p"]),
           "gbm_oof_log_loss_platt": log_loss(oof["y"], platt.predict(oof["p"])),
           "tune_years": TUNE_YEARS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    gb.head(25).to_csv(Path(path).with_name("gbm_tuning_top.csv"), index=False)
    lr.to_csv(Path(path).with_name("lr_tuning.csv"), index=False)
    return out
