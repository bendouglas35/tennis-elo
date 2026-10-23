"""Automated leakage checks for the feature tables.

Both checks work on an in-memory copy of `matches` from a built database, modify it, rebuild every
derived table with `derive`, and compare `model_features`.

* truncation: delete every match after a cutoff. Features of the remaining matches must not change,
  i.e. a feature never depends on the future.
* perturbation: overwrite the in-match stats of a sample of late matches from one week (repeated
  for several weeks). Their own features, and the features of everything at or before the earliest
  perturbed match, must not change; features of later matches must change (otherwise the test
  would be vacuous). Each sample holds at most one match per player and per event, and all its
  matches start in the same week, because perturbed matches that share a player or event, or that
  differ in week (court pace uses tour- and surface-wide means over earlier data), legitimately
  affect each other.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .build import derive

IN_MATCH_COLS = [
    "minutes", "games_w", "games_l", "sets_w", "sets_l",
    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt", "w_1stIn", "l_1stIn",
    "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon", "w_SvGms", "l_SvGms",
    "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
]


def copy_matches(src_db: str, **derive_kwargs):
    """In-memory connection holding a copy of `matches` and everything derived from it."""
    con = duckdb.connect()
    con.execute(f"attach '{src_db}' as src (read_only)")
    con.execute("create table matches as select * from src.matches")
    con.execute("detach src")
    derive(con, **derive_kwargs)
    return con


def _features(con) -> pd.DataFrame:
    return con.execute("select * from model_features order by match_id").df().set_index("match_id")


def mismatched_rows(a: pd.DataFrame, b: pd.DataFrame, rows) -> int:
    """Number of rows, among `rows`, where any feature column differs between frames a and b."""
    a, b = a.loc[rows], b.loc[rows]
    bad = np.zeros(len(a), dtype=bool)
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.kind in "fc":
            close = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-9)
            same = ((x.isna() & y.isna()) | close).to_numpy()
        else:
            same = ((x == y) | (x.isna() & y.isna())).to_numpy()
        bad |= ~same
    return int(bad.sum())


def truncation_check(src_db: str, quantiles=(0.3, 0.6, 0.85), **derive_kwargs) -> dict:
    full = _features(copy_matches(src_db, **derive_kwargs))
    ords = np.sort(full["ord_num"].to_numpy())
    cases = []
    for q in quantiles:
        cutoff = int(ords[int(q * (len(ords) - 1))])
        con = copy_matches(src_db, **derive_kwargs)
        con.execute(f"delete from matches where ord_num > {cutoff}")
        derive(con, **derive_kwargs)
        trunc = _features(con)
        con.close()
        rows = trunc.index.intersection(full.index)
        cases.append({"quantile": q, "rows_compared": len(rows),
                      "rows_changed": mismatched_rows(full, trunc, rows)})
    return {"cases": cases,
            "ok": all(c["rows_changed"] == 0 and c["rows_compared"] > 0 for c in cases)}


def thin_sample(base: pd.DataFrame, candidates: np.ndarray, n: int, seed: int) -> list:
    """Random matches, at most one per player and per event."""
    rng = np.random.default_rng(seed)
    seen_players, seen_events, keep = set(), set(), []
    for mid in rng.permutation(candidates):
        r = base.loc[mid]
        if r["player_a_id"] in seen_players or r["player_b_id"] in seen_players \
                or r["event_id"] in seen_events:
            continue
        seen_players.update([r["player_a_id"], r["player_b_id"]])
        seen_events.add(r["event_id"])
        keep.append(mid)
        if len(keep) == n:
            break
    return keep


def perturbation_check(src_db: str, n: int = 10, seed: int = 0, rounds: int = 6,
                       min_sample: int = 3, **derive_kwargs) -> dict:
    """Overwrite in-match stats of up to `n` matches that all start in the same week, `rounds` times.

    One week per round, at most one match per player and per event: pace and form share tour-wide
    and surface-wide means over *earlier* data, so perturbed matches in different weeks would
    legitimately change each other's rows. Within one week (and one event / player each) they cannot,
    so a perturbed match's own row and everything at or before it must stay identical, while later
    rows must move (otherwise the test is vacuous).
    """
    con = duckdb.connect()
    con.execute(f"attach '{src_db}' as src (read_only)")
    con.execute("create table matches_orig as select * from src.matches")
    con.execute("detach src")
    con.execute("create table matches as select * from matches_orig")
    derive(con, **derive_kwargs)
    base = _features(con)

    late = base[base["ord_num"] >= base["ord_num"].quantile(0.6)]
    dates = late["tourney_date"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    out = {"rounds": 0, "perturbed": 0, "own_rows_changed": 0, "earlier_rows_compared": 0,
           "earlier_rows_changed": 0, "later_rows_changed": 0}
    # Among 60 random late weeks, use the ones that allow the most independent matches.
    weeks = []
    for k, date in enumerate(rng.permutation(dates)[:60]):
        cands = late.index[late["tourney_date"] == date].to_numpy()
        sample = thin_sample(base, cands, n, seed + k)
        if len(sample) >= min_sample:
            weeks.append(sample)
    weeks.sort(key=len, reverse=True)
    for sample in weeks[:rounds]:
        con.register("sample_ids", pd.DataFrame({"match_id": sample}))
        con.execute("create or replace table matches as select * from matches_orig")
        sets = ", ".join(f"{c} = coalesce({c}, 0) + 13" for c in IN_MATCH_COLS)
        con.execute(f"update matches set {sets} where match_id in (select match_id from sample_ids)")
        derive(con, **derive_kwargs)
        pert = _features(con)

        sample_idx = pd.Index(sample)
        first_ord = int(base.loc[sample_idx, "ord_num"].min())
        before = base.index[base["ord_num"] <= first_ord]
        later = base.index[(base["ord_num"] > first_ord) & ~base.index.isin(sample_idx)]
        out["rounds"] += 1
        out["perturbed"] += len(sample_idx)
        out["own_rows_changed"] += mismatched_rows(base, pert, sample_idx)
        out["earlier_rows_compared"] += len(before)
        out["earlier_rows_changed"] += mismatched_rows(base, pert, before)
        out["later_rows_changed"] += mismatched_rows(base, pert, later)  # sensitivity: must be > 0
    con.close()
    out["ok"] = (out["own_rows_changed"] == 0 and out["earlier_rows_changed"] == 0
                 and out["later_rows_changed"] > 0 and out["perturbed"] > 0 and out["rounds"] > 0)
    return out
