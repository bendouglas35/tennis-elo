"""Validation (2022) and frozen-test reports for v0.4: tables with clustered bootstrap intervals,
calibration, SHAP, error slices and the pace/feature-group ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import TEST_START_YEAR, TRAIN_END_YEAR, VAL_YEAR
from . import dataset, models
from .experiments import CONFIG_PATH, dev_frame
from .metrics import (accuracy_vec, brier_vec, cluster_bootstrap, log_loss, log_loss_vec)

FIG_DIR = Path("docs/figures")
HEADLINE = ["Rank baseline", "Elo (tuned)", "LR: rank + Elo", "LR: all features, no pace",
            "LR: all features", "GBM: all features", "GBM: all features, recalibrated",
            "Average of LR and GBM"]


def load_config(path=CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def apply_slope(p, slope):
    return 1.0 / (1.0 + np.exp(-slope * models.logit(np.asarray(p))))


def fit_predict(tr: pd.DataFrame, ev: pd.DataFrame, cfg: dict):
    """Fit every headline model on `tr`, predict `ev`. Returns (predictions, fitted GBM, fitted LR)."""
    full = dataset.columns_for(dataset.FULL_GROUPS)
    base = dataset.columns_for(dataset.BASE_GROUPS)
    P = {}
    P["Rank baseline"] = models.LRModel(["d_log_rank"], C=1e3).fit(tr).predict(ev)
    P["Elo (tuned)"] = ev["p_elo"].to_numpy()
    P["LR: rank + Elo"] = models.LRModel(["d_log_rank", "d_elo_blend"], C=1e3).fit(tr).predict(ev)
    P["LR: all features, no pace"] = models.LRModel(base, cfg["lr_C"]).fit(tr).predict(ev)
    lr = models.LRModel(full, cfg["lr_C"]).fit(tr)
    P["LR: all features"] = lr.predict(ev)
    gbm = models.GBMModel(full, cfg["gbm_n_estimators"], **cfg["gbm_params"]).fit(tr)
    raw = gbm.predict(ev)
    P["GBM: all features"] = raw
    P["GBM: all features, recalibrated"] = apply_slope(raw, cfg["gbm_platt_slope"])
    P["Average of LR and GBM"] = 0.5 * (P["LR: all features"] + P["GBM: all features, recalibrated"])
    return P, gbm, lr


def summarize(P: dict, ev: pd.DataFrame, names=None, n_boot: int = 2000) -> dict:
    names = names or list(P)
    y = ev["y"].to_numpy()
    values, diffs = {}, []
    for k in names:
        values[f"ll:{k}"] = log_loss_vec(y, P[k])
        values[f"br:{k}"] = brier_vec(y, P[k])
        values[f"ac:{k}"] = accuracy_vec(y, P[k])
    for k in names:
        for ref in ("Elo (tuned)", "Rank baseline"):
            if k != ref and ref in names:
                diffs.append((f"ll:{k}", f"ll:{ref}"))
    for a, b in (("GBM: all features, recalibrated", "LR: all features"),
                 ("LR: all features", "LR: all features, no pace"),
                 ("Average of LR and GBM", "Elo (tuned)")):
        if a in names and b in names:
            diffs.append((f"ll:{a}", f"ll:{b}"))
    boot = cluster_bootstrap(ev["tourney_id"].to_numpy(), values, sorted(set(diffs)), n_boot=n_boot)
    return {"n": len(ev), "boot": boot, "names": names}


def table_md(s: dict) -> str:
    def f(t, d=4):
        return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"

    b = s["boot"]
    rows = ["| Model | Log loss (95% CI) | Brier | Accuracy | Δ log loss vs Elo | Δ vs rank baseline |",
            "|---|---|---|---|---|---|"]
    for k in s["names"]:
        ve = b.get(f"ll:{k} - ll:Elo (tuned)")
        vr = b.get(f"ll:{k} - ll:Rank baseline")
        rows.append(f"| {k} | {f(b[f'll:{k}'])} | {f(b[f'br:{k}'])} | {f(b[f'ac:{k}'], 3)} | "
                    f"{f(ve) if ve else ''} | {f(vr) if vr else ''} |")
    return "\n".join(rows)


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Symmetric reliability table: every match counted from both players' point of view."""
    P = np.concatenate([p, 1 - p]); Y = np.concatenate([y, 1 - y])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(P, edges) - 1, 0, bins - 1)
    rows = [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int((idx == i).sum()),
             "predicted": float(P[idx == i].mean()), "observed": float(Y[idx == i].mean())}
            for i in range(bins) if (idx == i).any()]
    return pd.DataFrame(rows)


def ece(y, p, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    return float((r["n"] * (r["predicted"] - r["observed"]).abs()).sum() / r["n"].sum())


def plot_calibration(y, P: dict, path, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    for k, p in P.items():
        r = reliability(y, p)
        ax.plot(r["predicted"], r["observed"], marker="o", ms=4, lw=1.4, label=f"{k} (ECE {ece(y, p):.3f})")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def shap_summary(gbm, ev: pd.DataFrame, path_bar, path_swarm, max_rows: int = 2500) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    X = ev[gbm.columns].iloc[:max_rows]
    sv = shap.TreeExplainer(gbm.m_).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(axis=0), index=gbm.columns).sort_values(ascending=False)
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("GBM feature importance (validation matches)", fontsize=10)
    fig.tight_layout()
    Path(path_bar).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_bar, dpi=140)
    plt.close(fig)
    plt.figure(figsize=(7, 6))
    shap.summary_plot(sv, X, max_display=15, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(path_swarm, dpi=140)
    plt.close()
    return imp


def slices(ev: pd.DataFrame, p_model, p_elo, p_rank) -> pd.DataFrame:
    y = ev["y"].to_numpy()
    df = ev.assign(_m=log_loss_vec(y, p_model), _e=log_loss_vec(y, p_elo), _r=log_loss_vec(y, p_rank),
                   _a=accuracy_vec(y, p_model))
    gap = df["d_log_rank"].abs()
    df["rank gap"] = pd.cut(gap, [-0.01, 0.3, 0.7, 1.2, 9], labels=["<0.3", "0.3-0.7", "0.7-1.2", ">1.2"])
    df["min Elo matches"] = pd.cut(df["elo_min_n"], [-1, 20, 100, 1e9], labels=["<20", "20-100", ">100"])
    df["pace"] = pd.cut(df["pace_static"], [-9, -0.2, 0.2, 9], labels=["slow", "medium", "fast"])
    df["surface"] = df["surface"].fillna("unknown")
    out = []
    for col in ("surface", "tourney_level", "round", "best_of", "rank gap", "min Elo matches", "pace"):
        for key, g in df.groupby(col, observed=True):
            if len(g) >= 40:
                out.append({"slice": f"{col} = {key}", "n": len(g), "accuracy": g["_a"].mean(),
                            "log loss": g["_m"].mean(), "Elo log loss": g["_e"].mean(),
                            "rank log loss": g["_r"].mean(), "model - Elo": g["_m"].mean() - g["_e"].mean()})
    return pd.DataFrame(out)


def biggest_misses(ev: pd.DataFrame, p, k: int = 10) -> pd.DataFrame:
    y = ev["y"].to_numpy()
    wp = np.where(y == 1, p, 1 - p)                    # probability given to the actual winner
    d = ev.assign(p_winner=wp).sort_values("p_winner").head(k)
    return d[["tourney_name", "round", "winner_name", "loser_name", "p_winner"]].reset_index(drop=True)


# ---- rolling-origin ablation ---------------------------------------------------------------

ABLATION_YEARS = [2018, 2019, 2020, 2021, 2022]
PACE_STEPS = [
    ("surface only (no indoor, altitude or pace)", dataset.BASE_GROUPS),
    ("+ indoor, altitude", dataset.BASE_GROUPS + ["indoor", "altitude"]),
    ("+ static pace", dataset.BASE_GROUPS + ["indoor", "altitude", "pace_static"]),
    ("+ running pace (full)", dataset.FULL_GROUPS),
]
DROPS = ["elo", "form", "fatigue", "rank", "entry", "bio"]


def ablation(df: pd.DataFrame, cfg: dict, kind: str, n_boot: int = 2000, cache: dict | None = None):
    """Rolling-origin (train < year) evaluation over 2018-2022 for nested feature-group sets."""
    def run(groups):
        cols = dataset.columns_for(groups)
        make = (lambda: models.LRModel(cols, cfg["lr_C"])) if kind == "LR" else \
               (lambda: models.GBMModel(cols, cfg["gbm_n_estimators"], **cfg["gbm_params"]))
        return models.rolling_origin(df, make, ABLATION_YEARS)

    variants = [(n, g) for n, g in PACE_STEPS]
    variants += [(f"full without {d}", [x for x in dataset.FULL_GROUPS if x != d]) for d in DROPS]
    preds = {n: run(g) for n, g in variants}
    full = preds["+ running pace (full)"]
    values = {n: log_loss_vec(p["y"], p["p"]) for n, p in preds.items()}
    diffs = [(n, "+ running pace (full)") for n in values if n != "+ running pace (full)"]
    diffs += [("+ indoor, altitude", "surface only (no indoor, altitude or pace)"),
              ("+ static pace", "+ indoor, altitude"), ("+ running pace (full)", "+ static pace")]
    boot = cluster_bootstrap(full["tourney_id"].to_numpy(), values, diffs, n_boot=n_boot)
    rows = []
    for n, _ in variants:
        r = {"variant": n, "log loss": boot[n]}
        if n != "+ running pace (full)":
            r["Δ vs full"] = boot[f"{n} - + running pace (full)"]
        rows.append(r)
    steps = []
    for a, b in diffs[-3:]:
        steps.append({"step": a, "Δ vs previous step": boot[f"{a} - {b}"]})
    return rows, steps, len(full)


def ablation_md(rows, steps, n, kind) -> str:
    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
    g = lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"
    out = [f"**{kind}**, rolling origin 2018-2022 (n = {n:,}); each year is predicted by a model trained on "
           "all earlier years.", "", "| Feature set | Log loss (95% CI) | Δ vs full model |", "|---|---|---|"]
    for r in rows:
        out.append(f"| {r['variant']} | {g(r['log loss'])} | {f(r['Δ vs full']) if 'Δ vs full' in r else ''} |")
    out += ["", "Pace steps, each against the previous one:", "",
            "| Step | Δ log loss (95% CI) |", "|---|---|"]
    out += [f"| {s['step']} | {f(s['Δ vs previous step'])} |" for s in steps]
    return "\n".join(out)


# ---- drivers --------------------------------------------------------------------------------

def _fmt_slices(sl: pd.DataFrame) -> str:
    lines = ["| Slice | n | Accuracy | Log loss | Elo log loss | Rank log loss | Model − Elo |", "|---|---|---|---|---|---|---|"]
    for r in sl.itertuples():
        lines.append(f"| {r.slice} | {r.n} | {r.accuracy:.3f} | {r._4:.4f} | {r._5:.4f} | {r._6:.4f} | {r._7:+.4f} |")
    return "\n".join(lines)


def _yearly(P: dict, ev: pd.DataFrame, names) -> str:
    lines = ["| Year | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for yr, g in ev.groupby("yr"):
        idx = ev.index.get_indexer(g.index)
        lines.append(f"| {yr} | {len(g)} | " + " | ".join(
            f"{log_loss(g['y'], np.asarray(P[k])[idx]):.4f}" for k in names) + " |")
    return "\n".join(lines)


def run_validation(db_path, out_dir="docs", n_boot: int = 2000, with_ablation: bool = True) -> dict:
    out_dir = Path(out_dir)
    df = dev_frame(db_path)
    cfg = load_config(out_dir / "model_config.json")
    tr, ev = df[df["yr"] <= TRAIN_END_YEAR], df[df["yr"] == VAL_YEAR].reset_index(drop=True)
    P, gbm, lr = fit_predict(tr, ev, cfg)
    s = summarize(P, ev, HEADLINE, n_boot)
    y = ev["y"].to_numpy()
    plot_calibration(y, {k: P[k] for k in ("Elo (tuned)", "LR: all features", "GBM: all features",
                                           "GBM: all features, recalibrated")},
                     FIG_DIR / "calibration_validation.png", f"Validation {VAL_YEAR} (n = {len(ev):,})")
    imp = shap_summary(gbm, ev, FIG_DIR / "shap_bar_validation.png", FIG_DIR / "shap_swarm_validation.png")
    sl = slices(ev, P["GBM: all features, recalibrated"], P["Elo (tuned)"], P["Rank baseline"])
    miss = biggest_misses(ev, P["GBM: all features, recalibrated"])
    coef = lr.coef().sort_values(key=np.abs, ascending=False)
    rel = {k: reliability(y, P[k]).to_dict("records") for k in ("Elo (tuned)", "LR: all features",
                                                                 "GBM: all features", "GBM: all features, recalibrated")}
    eces = {k: ece(y, P[k]) for k in HEADLINE}
    md = [f"## Validation {VAL_YEAR} (models fitted on {dataset.TRAIN_FIRST_YEAR}-{TRAIN_END_YEAR}, n = {len(ev):,})", "",
          table_md(s), "", "Expected calibration error (10 bins): " +
          ", ".join(f"{k} {v:.3f}" for k, v in eces.items()), "",
          f"GBM recalibration slope (fitted on 2018-2021 out-of-fold predictions): {cfg['gbm_platt_slope']:.3f}", "",
          "### Error slices (GBM recalibrated)", "", _fmt_slices(sl), "",
          "### Most confident misses (GBM recalibrated)", "", miss.to_markdown(floatfmt=".3f"), "",
          "### Logistic regression: largest standardised coefficients", "",
          coef.head(15).to_frame("coefficient").to_markdown(floatfmt=".3f"), "",
          "Pace interactions (standardised):", "",
          coef[[c for c in coef.index if "pace" in c or "alt" in c or "indoor" in c]].to_frame("coefficient").to_markdown(floatfmt=".3f"),
          "", "### GBM mean |SHAP| (top 15)", "", imp.head(15).to_frame("mean |SHAP|").to_markdown(floatfmt=".4f")]
    res = {"summary": {k: v for k, v in s["boot"].items()}, "n": s["n"], "ece": eces,
           "shap": imp.head(25).to_dict(), "lr_coef": coef.to_dict()}
    if with_ablation:
        for kind in ("LR", "GBM"):
            rows, steps, n = ablation(df, cfg, kind, n_boot)
            md += ["", f"### Ablation: {kind}", "", ablation_md(rows, steps, n, kind)]
            res[f"ablation_{kind}"] = {"rows": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]}
    (out_dir / "v04_validation.md").write_text("\n".join(md) + "\n")
    (out_dir / "v04_validation.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run_final_test(db_path, out_dir="docs", force: bool = False, n_boot: int = 2000) -> dict:
    """THE frozen-test milestone. Hyperparameters, Elo settings and feature sets are fixed; models are
    refitted on 2003-2022 (train + validation) and scored once on 2023 onwards."""
    out_dir = Path(out_dir)
    marker = out_dir / "test_results.json"
    if marker.exists() and not force:
        raise SystemExit(f"{marker} exists: the frozen test set was already evaluated. "
                         "Re-running means you are tuning on it; pass --force only to reproduce.")
    cfg = load_config(out_dir / "model_config.json")
    df = dataset.load(db_path, max_year=9999)
    tr, ev = df[df["yr"] <= VAL_YEAR], df[df["yr"] >= TEST_START_YEAR].reset_index(drop=True)
    assert tr["yr"].max() < ev["yr"].min()
    P, gbm, lr = fit_predict(tr, ev, cfg)
    s = summarize(P, ev, HEADLINE, n_boot)
    y = ev["y"].to_numpy()
    plot_calibration(y, {k: P[k] for k in ("Elo (tuned)", "LR: all features", "GBM: all features, recalibrated")},
                     FIG_DIR / "calibration_test.png", f"Frozen test {TEST_START_YEAR}+ (n = {len(ev):,})")
    sl = slices(ev, P["GBM: all features, recalibrated"], P["Elo (tuned)"], P["Rank baseline"])
    eces = {k: ece(y, P[k]) for k in HEADLINE}
    names3 = ["Rank baseline", "Elo (tuned)", "LR: all features", "GBM: all features, recalibrated"]
    md = [f"## Frozen test {TEST_START_YEAR}+ (models fitted on {dataset.TRAIN_FIRST_YEAR}-{VAL_YEAR}, n = {len(ev):,})", "",
          table_md(s), "", "Expected calibration error: " + ", ".join(f"{k} {v:.3f}" for k, v in eces.items()), "",
          "### By year (log loss)", "", _yearly(P, ev, names3), "", "### Error slices (GBM recalibrated)", "", _fmt_slices(sl)]
    res = {"n": s["n"], "summary": s["boot"], "ece": eces, "years": sorted(map(int, ev["yr"].unique())),
           "config": cfg}
    out_dir.mkdir(exist_ok=True)
    (out_dir / "test_results.md").write_text("\n".join(md) + "\n")
    marker.write_text(json.dumps(res, indent=1, default=float))
    return res


# ---- v0.5: do the four improvements help? ----------------------------------------------------

V05_STEPS = [
    ("v0.4 feature set", False, dataset.FULL_GROUPS),
    ("+ Elo learns from qualifying / Challenger results", True, dataset.FULL_GROUPS),
    ("+ rating trend (90 / 365 days)", True, dataset.FULL_GROUPS + ["trend"]),
    ("+ Glicko uncertainty", True, dataset.FULL_GROUPS + ["trend", "glicko"]),
    ("+ serve/return point model", True, dataset.FULL_GROUPS + ["trend", "glicko", "markov"]),
]


def run_v05_ablation(db_path, out_dir="docs", n_boot: int = 2000, steps=V05_STEPS, tag="v05") -> dict:
    """Rolling-origin comparison (2018-2022, train < year) of the v0.5 feature steps, in order.

    Development years only: nothing here loads 2023+. Each step is compared with the previous one
    and with the v0.4 feature set, for logistic regression and LightGBM at the v0.4 hyperparameters.
    """
    out_dir = Path(out_dir)
    cfg = load_config(out_dir / "model_config.json")
    frames = {False: dataset.load(db_path, VAL_YEAR, ext=False), True: dataset.load(db_path, VAL_YEAR, ext=True)}
    for f in frames.values():
        dataset.assert_no_test_years(f)
    res, md = {}, ["## v0.5 feature steps (rolling origin 2018-2022; each year predicted from earlier years)", ""]
    for kind in ("LR", "GBM"):
        preds = {}
        for name, ext, groups in steps:
            cols = dataset.columns_for(groups)
            make = (lambda c=cols: models.LRModel(c, cfg["lr_C"])) if kind == "LR" else \
                   (lambda c=cols: models.GBMModel(c, cfg["gbm_n_estimators"], **cfg["gbm_params"]))
            r = models.rolling_origin(frames[ext], make, ABLATION_YEARS)
            preds[name] = r.set_index("match_id")
            print(f"  {kind}: {name} done", flush=True)
        ids = preds[steps[0][0]].index
        for n in preds:
            ids = ids.intersection(preds[n].index)
        base = preds[steps[0][0]].loc[ids]
        values = {n: log_loss_vec(base["y"], preds[n].loc[ids, "p"]) for n in preds}
        v22 = (base["yr"] == VAL_YEAR).to_numpy()
        names = list(values)
        diffs = [(names[i], names[i - 1]) for i in range(1, len(names))] + \
                [(n, names[0]) for n in names[2:]]
        boot = cluster_bootstrap(base["tourney_id"].to_numpy(), values, diffs, n_boot=n_boot)
        boot22 = cluster_bootstrap(base["tourney_id"].to_numpy()[v22], {n: v[v22] for n, v in values.items()},
                                   [(names[i], names[i - 1]) for i in range(1, len(names))], n_boot=n_boot)
        f = lambda t, d=4: f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"
        g = lambda t, d=4: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
        md += [f"### {kind} (n = {len(ids):,} matches; validation year alone n = {int(v22.sum()):,})", "",
               "| Feature set | Log loss 2018-22 | Δ vs previous step | Δ vs v0.4 | Validation 2022 log loss | Δ vs previous (2022 only) |",
               "|---|---|---|---|---|---|"]
        for i, n in enumerate(names):
            md.append(f"| {n} | {g(boot[n])} | {f(boot[f'{n} - {names[i-1]}']) if i else ''} | "
                      f"{f(boot[f'{n} - {names[0]}']) if i > 1 else ''} | {g(boot22[n])} | "
                      f"{f(boot22[f'{n} - {names[i-1]}']) if i else ''} |")
        md.append("")
        res[kind] = {n: {"all": list(boot[n]), "val2022": list(boot22[n])} for n in names}
        res[kind]["steps"] = {f"{names[i]} - {names[i-1]}": list(boot[f"{names[i]} - {names[i-1]}"])
                              for i in range(1, len(names))}
    (out_dir / f"{tag}_ablation.md").write_text("\n".join(md) + "\n")
    (out_dir / f"{tag}_ablation.json").write_text(json.dumps(res, indent=1, default=float))
    return res
