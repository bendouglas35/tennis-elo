"""Tune the court-pace smoothing parameters on training years only (never validation or test).

Static pace is scored on how well it predicts each edition's realised mean composite (weighted by
matches); running pace on how well it predicts the composite of each later-round match. Squared
error on noisy targets, so only differences between settings matter, not the absolute level.
"""
from __future__ import annotations

import itertools

import duckdb
import pandas as pd

from . import TRAIN_END_YEAR
from .pace import PaceParams, final_sql, static_sql

FIRST_SCORED_YEAR = 2004      # earlier years have almost no history to pool
BIG = 1e9                     # "infinite" shrinkage = fall back to the next level


def _workspace(db_path: str):
    con = duckdb.connect()
    con.execute(f"attach '{db_path}' as src (read_only)")
    for t in ("pace_keys", "pace_match", "pace_edition", "pace_tour_mean", "altitude"):
        con.execute(f"create table {t} as select * from src.{t}")
    con.execute("create table matches as select match_id, tourney_id, event_id, ord_num, is_rateable "
                "from src.matches")
    con.execute("detach src")
    return con


def tune_static(con, grid=None) -> pd.DataFrame:
    grid = grid or itertools.product((0.4, 0.6, 0.8, 1.0), (3, 5, 8), (10, 25, 50, 100, 200, 400, BIG))
    rows = []
    for rho, k, m in grid:
        con.execute(static_sql(PaceParams(rho=rho, editions=k, m_static=m)))
        err = con.execute(f"""
            select sum(e.n * pow(e.s / e.n - st.pace_static, 2)) / sum(e.n), sum(e.n)
            from pace_edition e join pace_static st using (tourney_id)
            where e.n >= 5 and e.yr between {FIRST_SCORED_YEAR} and {TRAIN_END_YEAR}""").fetchone()
        rows.append({"rho": rho, "editions": k, "m_static": m, "mse": err[0], "matches": err[1]})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def tune_running(con, static: PaceParams, grid=(5, 10, 20, 30, 60, 120, BIG)) -> pd.DataFrame:
    rows = []
    for m in grid:
        p = PaceParams(static.rho, static.editions, static.m_static, m_running=m)
        con.execute(static_sql(p))
        con.execute(final_sql(p))
        err = con.execute(f"""
            select avg(pow(pm.comp - mp.pace_running, 2)), count(*)
            from pace_match pm join match_pace mp using (match_id)
            where mp.pace_n_running > 0 and pm.yr between {FIRST_SCORED_YEAR} and {TRAIN_END_YEAR}""").fetchone()
        rows.append({"m_running": m, "mse": err[0], "matches": err[1]})
    return pd.DataFrame(rows)


def run(db_path: str) -> dict:
    con = _workspace(db_path)
    s = tune_static(con)
    best = s.iloc[0]
    static = PaceParams(rho=float(best.rho), editions=int(best.editions), m_static=float(best.m_static))
    r = tune_running(con, static)
    return {"static": s, "running": r, "best_static": static,
            "best_running": float(r.sort_values("mse").iloc[0].m_running)}
