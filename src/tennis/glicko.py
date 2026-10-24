"""Glicko-1 ratings with a per-match sequential update (rating deviation = uncertainty).

Unlike Elo's fixed K, a player's step size depends on how sure we are about him: the deviation grows
with inactivity (c per sqrt of 30-day period) and shrinks with every match. A new player, or one back
from a long layoff, therefore moves fast, and a settled veteran moves slowly. Matches that share an
order key are simultaneous: every pre-match state in the group is read before any update.
Auxiliary (qualifying / Challenger) rows update ratings when `use_aux` is set and are never scored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

Q = math.log(10.0) / 400.0
BASE, RD0 = 1500.0, 350.0


@dataclass(frozen=True)
class GlickoConfig:
    c: float = 35.0          # rating-points of deviation growth per sqrt(30 days) of inactivity
    rd_min: float = 40.0     # deviation never falls below this
    use_aux: bool = False
    aux_weight: float = 1.0  # auxiliary matches move ratings by this factor (via a scaled outcome gap)


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi ** 2))


def run_glicko(df: pd.DataFrame, cfg: GlickoConfig) -> pd.DataFrame:
    """Pre-match g (rating) and rd (deviation) of both players for every standard match, plus the
    winner's win probability p_w that accounts for both deviations."""
    if "aux" not in df.columns:
        df = df.assign(aux=False)
    if not cfg.use_aux:
        df = df[~df["aux"].to_numpy(bool)].reset_index(drop=True)
    n = len(df)
    ord_ = df["ord_num"].to_numpy()
    day = df["day"].to_numpy()
    std, aux = df["standard"].to_numpy(bool), df["aux"].to_numpy(bool)
    win, los = df["winner_id"].tolist(), df["loser_id"].tolist()
    R, RD, last = {}, {}, {}
    out = {k: np.full(n, np.nan) for k in ("w_g", "w_rd", "l_g", "l_rd", "p_w")}

    def pre(p, d):
        r, rd = R.get(p, BASE), RD.get(p, RD0)
        if p in last and d > last[p]:
            rd = min(RD0, math.sqrt(rd * rd + cfg.c ** 2 * (d - last[p]) / 30.0))
        return r, rd

    bounds = np.flatnonzero(np.diff(ord_)) + 1
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [n]])
    for a, b in zip(starts, ends):
        pending = []
        for i in range(a, b):
            if not (std[i] or aux[i]):
                continue
            d = int(day[i])
            (rw, dw), (rl, dl) = pre(win[i], d), pre(los[i], d)
            if std[i]:
                out["w_g"][i], out["w_rd"][i], out["l_g"][i], out["l_rd"][i] = rw, dw, rl, dl
                out["p_w"][i] = 1.0 / (1.0 + 10.0 ** (-_g(math.hypot(dw, dl)) * (rw - rl) / 400.0))
            pending.append((i, rw, dw, rl, dl))
        for i, rw, dw, rl, dl in pending:
            wgt = cfg.aux_weight if aux[i] else 1.0
            for me, rd_me, opp_r, opp_rd, score, who in ((win[i], dw, rl, dl, 1.0, "w"), (los[i], dl, rw, dw, 0.0, "l")):
                my_r = rw if who == "w" else rl
                g = _g(opp_rd)
                e = 1.0 / (1.0 + 10.0 ** (-g * (my_r - opp_r) / 400.0))
                d2 = 1.0 / (Q * Q * g * g * e * (1.0 - e))
                prec = 1.0 / (rd_me * rd_me) + 1.0 / d2
                new_rd = max(cfg.rd_min, math.sqrt(1.0 / prec))
                R[me] = my_r + wgt * Q / prec * g * (score - e)
                RD[me] = new_rd
        for i in range(a, b):
            last[win[i]] = last[los[i]] = int(day[i])

    res = pd.DataFrame(out, index=df.index)
    res.insert(0, "match_id", df["match_id"].to_numpy())
    return res[std].reset_index(drop=True)
