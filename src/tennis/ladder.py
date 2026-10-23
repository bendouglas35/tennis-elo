"""Elo ladder (v0.3): add one extension at a time, tune it on training years only, report it on the
2022 validation year with tournament-clustered bootstrap intervals.

Tuning window: 2008-2021 (earlier years are rating warm-up). Validation (2022) is only scored, never
used to choose a value. A step is *kept* only if it lowers the tuning-window log loss; otherwise the
row is still reported but the next step builds on the previous kept configuration.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace

import duckdb
import numpy as np
import pandas as pd

from . import TRAIN_END_YEAR, VAL_YEAR
from .baseline import fit_logistic, predict
from .elo import EloConfig, load_matches, run_elo
from .metrics import accuracy_vec, brier_vec, cluster_bootstrap, log_loss_vec

TUNE_FIRST_YEAR = 2008


class Workspace:
    """Loaded once: all rateable matches plus year / rank metadata for the standard ones."""

    def __init__(self, db_path):
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            self.df = load_matches(con)
            meta = con.execute("""select match_id, year(tourney_date) as yr, winner_rank, loser_rank
                                  from matches where is_rateable""").df()
        finally:
            con.close()
        std = self.df[self.df["standard"]][["match_id", "tourney_id"]].reset_index(drop=True)
        self.meta = std.merge(meta, on="match_id", how="left")
        assert len(self.meta) == int(self.df["standard"].sum())

    def pre(self, cfg: EloConfig) -> pd.DataFrame:
        res = run_elo(self.df, cfg)
        assert (res["match_id"].to_numpy() == self.meta["match_id"].to_numpy()).all()
        return res

    def tune_loss(self, cfg: EloConfig) -> float:
        p = self.pre(cfg)["p_w"].to_numpy()
        m = ((self.meta["yr"] >= TUNE_FIRST_YEAR) & (self.meta["yr"] <= TRAIN_END_YEAR)).to_numpy()
        return float(-np.log(np.clip(p[m], 1e-12, 1)).mean())

    def rank_baseline(self) -> pd.Series:
        """Winner-perspective win probability from a logistic on the log rank ratio, fitted on
        training years (<= 2021) with no intercept (symmetric)."""
        m = self.meta
        ok = m["winner_rank"].notna() & m["loser_rank"].notna() & (m["winner_rank"] != m["loser_rank"])
        ok = ok & (m["winner_rank"] > 0) & (m["loser_rank"] > 0)
        x = np.log(m["loser_rank"].astype(float).where(ok) / m["winner_rank"].astype(float).where(ok))
        tr = (ok & (m["yr"] <= TRAIN_END_YEAR)).to_numpy()
        xs = x[tr].to_numpy()
        beta = fit_logistic(np.concatenate([xs, -xs]), np.concatenate([np.ones(len(xs)), np.zeros(len(xs))]))
        return pd.Series(np.where(ok, predict(np.array([0.0, beta[1]]), x.fillna(0).to_numpy()), np.nan),
                         index=m.index)


def _grid(base: EloConfig, **axes):
    keys = list(axes)
    for vals in itertools.product(*axes.values()):
        yield replace(base, **dict(zip(keys, vals)))


STEPS = [
    ("Elo, fixed K", lambda b: _grid(b, k_mode=("fixed",), k_fixed=(16.0, 24.0, 32.0, 48.0, 64.0))),
    ("+ dynamic K", lambda b: _grid(b, k_mode=("dynamic",), k0=(150.0, 250.0, 400.0),
                                    k_shape=(0.2, 0.4, 0.6), k_floor=(0.0, 16.0, 20.0))),
    ("+ rating scale", lambda b: _grid(b, scale=(350.0, 400.0, 450.0))),
    ("+ best-of-5 conversion", lambda b: _grid(b, bo5=(True,))),
    ("+ surface blend", lambda b: _grid(b, surface_w=(0.25, 0.5, 0.75, 1.0))),
    ("+ layoff uncertainty", lambda b: _grid(b, layoff_grace=(60.0, 120.0, 180.0),
                                             layoff_tau=(180.0, 365.0, 730.0), layoff_boost=(0.0, 1.0, 2.0))),
    ("+ margin of victory", lambda b: _grid(b, mov=(0.5, 1.0, 2.0, 3.0))),
    ("+ retirement scaling", lambda b: _grid(b, retire_scale=(0.0, 0.5))),
]


def _changed(cfg: EloConfig, base: EloConfig) -> dict:
    a, b = asdict(cfg), asdict(base)
    return {k: v for k, v in a.items() if v != b[k]}


def run(db_path, n_boot: int = 2000, verbose: bool = True) -> dict:
    ws = Workspace(db_path)
    default = EloConfig()
    kept = replace(default, k_mode="fixed")      # starting point for the first step
    kept_loss = None
    rows = []
    for name, make in STEPS:
        cands = list(make(kept))
        scored = [(ws.tune_loss(c), c) for c in cands]
        loss, best = min(scored, key=lambda t: t[0])
        improved = kept_loss is None or loss < kept_loss - 1e-6
        if verbose:
            print(f"{name:<26} tune ll {loss:.5f} ({'kept' if improved else 'no gain'}) "
                  f"{_changed(best, default)}", flush=True)
        rows.append({"step": name, "cfg": best if improved else kept, "tried": best,
                     "tune_loss": loss, "kept": improved})
        if improved:
            kept, kept_loss = best, loss
        else:
            rows[-1]["tune_loss"] = kept_loss

    base_p = ws.rank_baseline()
    val = (ws.meta["yr"] == VAL_YEAR) & base_p.notna()
    n_val_all = int((ws.meta["yr"] == VAL_YEAR).sum())
    y = np.ones(int(val.sum()))
    groups = ws.meta.loc[val, "tourney_id"].to_numpy()
    preds = {"Rank baseline": base_p[val].to_numpy()}
    for r in rows:
        r["p_val"] = ws.pre(r["tried"])["p_w"].to_numpy()[val.to_numpy()]
        preds[r["step"]] = r["p_val"]
    names = list(preds)
    values = {}
    for k in names:
        values[f"ll:{k}"] = log_loss_vec(y, preds[k])
        values[f"br:{k}"] = brier_vec(y, preds[k])
        values[f"ac:{k}"] = accuracy_vec(y, preds[k])
    diffs, prev = [], "Rank baseline"
    for r in rows:
        diffs.append((f"ll:{r['step']}", f"ll:{prev}"))
        diffs.append((f"ll:{r['step']}", "ll:Rank baseline"))
        if r["kept"]:
            prev = r["step"]
    boot = cluster_bootstrap(groups, values, diffs, n_boot=n_boot, seed=0)

    table, prev = [], "Rank baseline"
    for k in names:
        row = {"model": k, "ll": boot[f"ll:{k}"], "brier": boot[f"br:{k}"], "acc": boot[f"ac:{k}"]}
        if k != "Rank baseline":
            r = next(r for r in rows if r["step"] == k)
            row["tune_loss"] = r["tune_loss"]
            row["kept"] = r["kept"]
            row["vs_prev"] = boot[f"ll:{k} - ll:{prev}"]
            row["vs_rank"] = boot[f"ll:{k} - ll:Rank baseline"]
            row["changed"] = _changed(r["tried"], default)
            if r["kept"]:
                prev = k
        table.append(row)
    return {"table": table, "final_cfg": kept, "n_val": int(val.sum()), "n_val_all": n_val_all,
            "kept_tune_loss": kept_loss}


def to_markdown(res: dict) -> str:
    def f(t, d=4):
        return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"

    lines = ["| Model | Val log loss (95% CI) | Val Brier | Val accuracy | Δ log loss vs previous kept | Δ vs rank baseline | Kept |",
             "|---|---|---|---|---|---|---|"]
    for r in res["table"]:
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["model"], f(r["ll"]), f(r["brier"]), f(r["acc"], 3),
            f(r["vs_prev"]) if "vs_prev" in r else "", f(r["vs_rank"]) if "vs_rank" in r else "",
            ("yes" if r["kept"] else "no") if "kept" in r else ""))
    return "\n".join(lines)


def dump(res: dict, path) -> None:
    out = {"final_cfg": asdict(res["final_cfg"]), "n_val": res["n_val"], "n_val_all": res["n_val_all"],
           "kept_tune_loss": res["kept_tune_loss"],
           "table": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                     for r in res["table"]]}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
