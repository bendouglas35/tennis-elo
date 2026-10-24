"""Sequential, leakage-safe Elo ratings with the extensions in docs/feature_plan.md.

Matches are processed in `ord_num` order (tournament week, then round). Matches that share an
order key are *simultaneous*: every pre-match rating in a group is read before any update from that
group is applied. Team events and other non-standard formats never update ratings and are never
scored, but they do count as activity (they reset the inactivity clock).

Variants (all switchable through `EloConfig`):
* fixed K, or dynamic K = k0 / (N + k_off) ** k_shape, where N is the effective number of matches;
* surface blend: rating used for prediction = (1 - w) * overall + w * surface rating;
* layoff: after `layoff_grace` days of inactivity N decays by exp(-(gap - grace) / tau), so a
  dynamic K rises again; `layoff_boost` additionally multiplies K by 1 + boost * (1 - decay), which
  also works with a fixed K;
* margin of victory: the update is scaled by 1 + mov * (2 * game share - 1);
* retirement: updates after a retirement are scaled by `retire_scale` (0 = ignored);
* best-of-five conversion: the best-of-three win probability implied by the rating gap is converted
  to a best-of-five probability through the set-win probability.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = 1500.0


@dataclass(frozen=True)
class EloConfig:
    k_mode: str = "dynamic"          # "fixed" | "dynamic"
    k_fixed: float = 32.0
    k0: float = 250.0
    k_off: float = 5.0
    k_shape: float = 0.4
    k_floor: float = 0.0             # dynamic K never drops below this
    scale: float = 400.0             # rating gap that means 10:1 odds
    bo5: bool = False
    surface_w: float = 0.0
    layoff_grace: float | None = None  # days; None = no layoff effect
    layoff_tau: float = 365.0
    layoff_boost: float = 0.0        # K multiplier 1 + boost * (1 - decay) after a layoff
    mov: float = 0.0
    retire_scale: float = 1.0
    aux_k: float = 0.0               # K scale for qualifying / Challenger results (0 = ignore them)
    rank_beta: float = 0.0           # new players start at 1500 + beta * ln(rank_ref / rank)
    rank_ref: float = 150.0


# set-win probability grid for the best-of-3 <-> best-of-5 conversion
_S = np.linspace(0.0, 1.0, 4001)
_P3 = _S ** 2 * (3 - 2 * _S)
_P5 = _S ** 3 * (10 - 15 * _S + 6 * _S ** 2)


def bo3_to_bo5(p3: float) -> float:
    """Best-of-5 win probability for a player whose best-of-3 win probability is p3."""
    return float(np.interp(p3, _P3, _P5))


def surface_key(surface):
    if surface is None or (isinstance(surface, float) and math.isnan(surface)):
        return None
    return "Hard" if surface == "Carpet" else surface


def _prob(diff: float, scale: float, best_of: int, bo5: bool) -> float:
    p = 1.0 / (1.0 + 10.0 ** (-diff / scale))
    return bo3_to_bo5(p) if (bo5 and best_of == 5) else p


def load_matches(con, aux: bool = False) -> pd.DataFrame:
    """All rateable matches (standard and team) in order, with the columns the engine needs.

    aux=True also appends qualifying / Challenger results (table `aux_matches`, see aux.py): they can
    update ratings but are never scored.
    """
    sql = """
        select match_id, tourney_id, ord_num, (tourney_date - date '1970-01-01')::int as day,
               surface, coalesce(best_of, 3) as best_of, winner_id, loser_id, games_w, games_l,
               coalesce(score_status = 'retired', false) as retired,
               not (is_team_event or is_nonstandard_format) as standard, false as aux,
               winner_rank, loser_rank
        from matches where is_rateable"""
    if aux:
        sql += """
        union all
        select match_id, tourney_id, ord_num, (tourney_date - date '1970-01-01')::int as day,
               surface, coalesce(best_of, 3), winner_id, loser_id, games_w, games_l,
               coalesce(retired, false), false, true, winner_rank, loser_rank
        from aux_matches"""
    return con.execute(sql + " order by ord_num, match_id").df()


def _start(cfg: EloConfig, rank) -> float:
    """Rating of a player seen for the first time. With rank seeding it depends on his ATP rank."""
    if cfg.rank_beta and rank == rank and rank > 0:          # rank == rank filters NaN
        r = BASE + cfg.rank_beta * (math.log(cfg.rank_ref) - math.log(rank))
        return min(max(r, BASE - 250.0), BASE + 350.0)
    return BASE


def run_elo(df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """Pre-match ratings for every standard match. Columns (w_ = winner, l_ = loser):
    r (overall), s (surface, falls back to overall), b (blend used for prediction), n (effective
    matches), gap (days since last match, NaN if none), tr90 / tr365 (overall rating minus the rating
    90 / 365 days earlier, NaN without history), plus p_w, the winner's pre-match win probability."""
    if "aux" not in df.columns:
        df = df.assign(aux=False)
    if cfg.aux_k == 0.0:
        df = df[~df["aux"].to_numpy(bool)].reset_index(drop=True)
    R, N, S, NS, last = {}, {}, {}, {}, {}
    hist_d, hist_r = {}, {}
    ord_ = df["ord_num"].to_numpy()
    cols = {c: df[c].to_numpy() for c in ("day", "best_of", "retired", "standard", "aux")}
    cols["games_w"] = df["games_w"].astype(float).to_numpy()
    cols["games_l"] = df["games_l"].astype(float).to_numpy()
    nan = np.full(len(df), np.nan)
    wrank = df["winner_rank"].astype(float).to_numpy() if "winner_rank" in df else nan
    lrank = df["loser_rank"].astype(float).to_numpy() if "loser_rank" in df else nan
    win = df["winner_id"].tolist()
    los = df["loser_id"].tolist()
    surf = [surface_key(s) for s in df["surface"].tolist()]
    n = len(df)
    out = {k: np.full(n, np.nan) for k in
           ("w_r", "w_s", "w_b", "w_n", "w_gap", "w_tr90", "w_tr365",
            "l_r", "l_s", "l_b", "l_n", "l_gap", "l_tr90", "l_tr365", "p_w")}
    grace, tau = cfg.layoff_grace, cfg.layoff_tau

    def decay(gap):
        if grace is None or gap is None or gap <= grace:
            return 1.0
        return math.exp(-(gap - grace) / tau)

    def kfun(n_eff):
        if cfg.k_mode == "fixed":
            return cfg.k_fixed
        return max(cfg.k_floor, cfg.k0 / (n_eff + cfg.k_off) ** cfg.k_shape)

    def rating_at(p, day):
        d = hist_d.get(p)
        if not d:
            return None
        k = bisect_right(d, day) - 1
        return hist_r[p][k] if k >= 0 else None

    bounds = np.flatnonzero(np.diff(ord_)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    wt_surface = cfg.surface_w
    for a, b in zip(starts, ends):
        pending = []
        for i in range(a, b):                     # 1) read every pre-match state in the group
            std, is_aux = cols["standard"][i], cols["aux"][i]
            if not (std or is_aux):
                continue
            w, l, sf, day = win[i], los[i], surf[i], int(cols["day"][i])
            snap = {}
            for tag, p, rk in (("w", w, wrank[i]), ("l", l, lrank[i])):
                gap = None if p not in last else day - last[p]
                d = decay(gap)
                r = R[p] if p in R else _start(cfg, rk)
                n_eff = N.get(p, 0.0) * d
                if sf is not None:
                    s = S.get((p, sf), r)          # first match on a surface starts at overall
                    ns = NS.get((p, sf), 0.0) * d
                else:
                    s, ns = r, 0.0
                bl = (1 - wt_surface) * r + wt_surface * s if sf is not None else r
                snap[tag] = (r, s, bl, n_eff, ns, 1.0 + cfg.layoff_boost * (1.0 - d))
                if std:
                    out[f"{tag}_r"][i], out[f"{tag}_s"][i], out[f"{tag}_b"][i] = r, s, bl
                    out[f"{tag}_n"][i] = n_eff
                    out[f"{tag}_gap"][i] = np.nan if gap is None else gap
                    for h in (90, 365):
                        old = rating_at(p, day - h)
                        out[f"{tag}_tr{h}"][i] = np.nan if old is None else r - old
            bo = int(cols["best_of"][i])
            if std:
                out["p_w"][i] = _prob(snap["w"][2] - snap["l"][2], cfg.scale, bo, cfg.bo5)
            pending.append((i, snap))
        for i, snap in pending:                   # 2) apply the updates
            w, l, sf = win[i], los[i], surf[i]
            if cols["retired"][i]:
                f = cfg.retire_scale
            elif cfg.mov and not (math.isnan(cols["games_w"][i]) or math.isnan(cols["games_l"][i])) \
                    and cols["games_w"][i] + cols["games_l"][i] > 0:
                gw, gl = cols["games_w"][i], cols["games_l"][i]
                f = 1.0 + cfg.mov * (2.0 * gw / (gw + gl) - 1.0)
            else:
                f = 1.0
            if cols["aux"][i]:
                f *= cfg.aux_k
            if f == 0.0:
                continue
            bo = int(cols["best_of"][i])
            day = int(cols["day"][i])
            (rw, sw, _, nw, nsw, kmw), (rl, sl, _, nl, nsl, kml) = snap["w"], snap["l"]
            e_o = _prob(rw - rl, cfg.scale, bo, cfg.bo5)
            R[w] = rw + kfun(nw) * kmw * f * (1 - e_o) if w not in R else R[w] + kfun(nw) * kmw * f * (1 - e_o)
            R[l] = rl - kfun(nl) * kml * f * (1 - e_o) if l not in R else R[l] - kfun(nl) * kml * f * (1 - e_o)
            N[w], N[l] = nw + 1, nl + 1
            for p in (w, l):
                hist_d.setdefault(p, []).append(day)
                hist_r.setdefault(p, []).append(R[p])
            if sf is not None:
                e_s = _prob(sw - sl, cfg.scale, bo, cfg.bo5)
                S[(w, sf)] = S.get((w, sf), sw) + kfun(nsw) * kmw * f * (1 - e_s)
                S[(l, sf)] = S.get((l, sf), sl) - kfun(nsl) * kml * f * (1 - e_s)
                NS[(w, sf)], NS[(l, sf)] = nsw + 1, nsl + 1
        for i in range(a, b):                     # 3) activity (team events and auxiliary matches too)
            day = int(cols["day"][i])
            last[win[i]] = day
            last[los[i]] = day

    res = pd.DataFrame(out, index=df.index)
    res.insert(0, "match_id", df["match_id"].to_numpy())
    return res[df["standard"].to_numpy()].reset_index(drop=True)


# Configuration chosen by the v0.3 ladder (tuned on 2008-2021 only; see docs/elo_ladder.md).
TUNED = EloConfig(k_mode="fixed", k_fixed=24.0, bo5=True, surface_w=0.5, layoff_grace=60.0,
                  layoff_tau=180.0, layoff_boost=1.0, mov=1.0)
