"""Modelling frame for v0.4: SQL features + Elo features, in symmetric A/B orientation.

Feature roles under the A <-> B mirror (swap players, flip the target):
* ANTI columns (`d_*`) change sign. All player-difference features, and every interaction of a
  symmetric context column with an antisymmetric one, are ANTI.
* SYM columns do not change (surface, level, round, pace, altitude, ...).
A column is only ever added through GROUPS, so ablations are explicit and the mirror is exact.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import TEST_START_YEAR, VAL_YEAR
from dataclasses import replace

from . import pointmodel
from .elo import TUNED, EloConfig, load_matches, run_elo
from .glicko import GlickoConfig, run_glicko

TRAIN_FIRST_YEAR = 2003          # earlier years are warm-up for ratings and form

LEVELS = ["G", "M", "500", "250", "F"]
ROUNDS = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "RR": 5, "SF": 6, "F": 7, "BR": 7}

_ANTI_FORM = [
    "d_f_first_in_pct", "d_f_first_won_pct", "d_f_second_won_pct", "d_f_ace_rate",
    "d_f_ace_allowed_rate", "d_f_df_rate", "d_f_sv_won_pct", "d_f_bp_saved_pct", "d_f_ret_won_pct",
    "d_f_bp_conv_pct", "d_f_win_rate", "d_f_game_win_pct", "d_f_dominance_ratio", "d_f_eff_stats",
    "d_fs_sv_won_pct", "d_fs_ret_won_pct", "d_fs_ace_rate", "d_fs_win_rate", "d_fs_game_win_pct",
    "d_fs_dominance_ratio", "d_fs_eff_stats", "d_f_n_stats",
]
GROUPS = {
    "rank": ["d_log_rank", "d_log_rank_points", "log_rank_sum"],
    "bio": ["d_age", "d_ht", "d_lefty"],
    "form": _ANTI_FORM,
    "fatigue": ["d_days_since_last", "d_career_matches", "d_tourney_matches_before",
                "d_tourney_minutes_before", "d_tourney_sets_before", "d_n_14d", "d_minutes_14d",
                "d_n_28d", "d_surface_matches_365d", "d_prev_retired"],
    "entry": ["d_seeded", "d_qualifier", "d_wildcard", "d_lucky_loser"],
    "elo": ["d_elo_blend", "d_elo_overall", "d_elo_surface", "d_elo_n", "d_elo_gap", "elo_min_n"],
    "context": ["surf_hard", "surf_clay", "surf_grass", "lvl_G", "lvl_M", "lvl_500", "lvl_250",
                "lvl_F", "round_ord", "best_of", "draw_size"],
    "indoor": ["indoor", "d_x_indoor_ace", "d_x_indoor_sv"],
    "altitude": ["altitude_km", "d_x_alt_ace"],
    "pace_static": ["pace_static", "d_x_pace_s_ace", "d_x_pace_s_sv", "d_x_pace_s_ret"],
    "pace_running": ["pace_running", "d_x_pace_r_ace", "d_x_pace_r_sv", "d_x_pace_r_ret"],
}
# v0.5 additions (only present when the frame is loaded with ext=True)
GROUPS.update({
    "trend": ["d_elo_tr90", "d_elo_tr365"],
    "glicko": ["d_glicko", "glicko_sd", "d_glicko_p"],
    "markov": ["d_markov"],
})
ALL_COLUMNS = [c for cols in GROUPS.values() for c in cols]
SYM = {"glicko_sd", "log_rank_sum", "elo_min_n", "surf_hard", "surf_clay", "surf_grass", "lvl_G", "lvl_M", "lvl_500",
       "lvl_250", "lvl_F", "round_ord", "best_of", "draw_size", "indoor", "altitude_km", "pace_static",
       "pace_running"}
ANTI = [c for c in ALL_COLUMNS if c not in SYM]
BASE_GROUPS = ["rank", "bio", "form", "fatigue", "entry", "elo", "context"]
FULL_GROUPS = BASE_GROUPS + ["indoor", "altitude", "pace_static", "pace_running"]
EXT_STEPS = ["trend", "glicko", "markov"]                  # v0.5 groups, added in this order

# Settings fixed by the v0.5 tuning (see docs/v05_improvements.md)
ELO_EXT = replace(TUNED, aux_k=0.75)
# v0.5 decision (docs/v05_improvements.md): keep only the aux-data Elo (ext=True) with the v0.4 feature
# groups. The trend, glicko and markov groups stay available but did not clear the pre-stated bar.
V05_GROUPS = None  # filled below once FULL_GROUPS is defined
GLICKO_TUNED = GlickoConfig(c=20.0, rd_min=60.0, use_aux=True)
MARKOV_PACE_K = 0.0          # serve-point shift per unit of pace, in units of the pace SD constant
MARKOV_SURFACE_MIX = 0.5     # weight of surface-specific serve/return form in the point model


def columns_for(groups) -> list:
    return [c for g in groups for c in GROUPS[g]]


def _sql(max_year: int) -> str:
    return f"""
    select mf.*, ms.rank_a, ms.rank_b, year(mf.tourney_date) as yr,
           pa.f_sv_won_pct as sv_a, pb.f_sv_won_pct as sv_b, pa.f_ret_won_pct as ret_a, pb.f_ret_won_pct as ret_b,
           pa.fs_sv_won_pct as ssv_a, pb.fs_sv_won_pct as ssv_b,
           pa.fs_ret_won_pct as sret_a, pb.fs_ret_won_pct as sret_b, tm.mean_sv
    from model_features mf join match_sides ms using (match_id)
    join player_features pa on pa.match_id = mf.match_id and pa.player_id = mf.player_a_id
    join player_features pb on pb.match_id = mf.match_id and pb.player_id = mf.player_b_id
    left join pace_tour_mean tm on tm.tourney_date = mf.tourney_date
    where not mf.is_team_event and not mf.is_nonstandard_format
      and year(mf.tourney_date) >= {TRAIN_FIRST_YEAR} and year(mf.tourney_date) <= {max_year}
    order by mf.ord_num, mf.match_id"""


def load(db_path, max_year: int, elo_cfg: EloConfig | None = None, ext: bool = False) -> pd.DataFrame:
    """Modelling frame for standard tour matches up to and including `max_year`.

    Validation work passes max_year=VAL_YEAR, so test years are never even loaded.
    ext=True is the v0.5 frame: ratings also learn from qualifying / Challenger results (if loaded),
    and the Glicko and point-model features are added.
    """
    elo_cfg = elo_cfg or (ELO_EXT if ext else TUNED)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(_sql(max_year)).df()
        has_aux = ext and con.execute("select count(*) from information_schema.tables "
                                      "where table_name = 'aux_matches'").fetchone()[0] > 0
        matches = load_matches(con, aux=has_aux)
        names = con.execute("select match_id, winner_name, loser_name, tourney_name from matches").df()
    finally:
        con.close()
    matches = matches[matches["ord_num"] <= int(df["ord_num"].max())]   # never feed ratings later matches
    elo = run_elo(matches, elo_cfg)
    df = df.merge(elo, on="match_id", how="left").merge(names, on="match_id", how="left")
    if ext:
        gl = run_glicko(matches, GLICKO_TUNED).rename(columns=lambda c: c if c == "match_id" else f"g_{c}")
        df = df.merge(gl, on="match_id", how="left")
    return add_features(df, ext=ext)


def add_features(df: pd.DataFrame, ext: bool = False) -> pd.DataFrame:
    df = df.copy()
    aw = df["a_wins"].astype(bool).to_numpy()
    for stat in ("r", "s", "b", "n", "gap"):
        w, l = df[f"w_{stat}"].to_numpy(float), df[f"l_{stat}"].to_numpy(float)
        df[f"{stat}_a"], df[f"{stat}_b"] = np.where(aw, w, l), np.where(aw, l, w)
    # orientation only picks whose pre-match column to read; the values themselves are per player
    df["p_elo"] = np.where(aw, df["p_w"], 1 - df["p_w"])
    df["d_elo_blend"] = df["b_a"] - df["b_b"]
    df["d_elo_overall"] = df["r_a"] - df["r_b"]
    df["d_elo_surface"] = df["s_a"] - df["s_b"]
    df["d_elo_n"] = df["n_a"] - df["n_b"]
    df["d_elo_gap"] = df["gap_a"] - df["gap_b"]
    df["elo_min_n"] = np.minimum(df["n_a"], df["n_b"])
    df["d_lefty"] = df["lefty_a"].astype(float) - df["lefty_b"].astype(float)
    for name, col in (("seeded", "is_seeded"), ("qualifier", "is_qualifier"),
                      ("wildcard", "is_wildcard"), ("lucky_loser", "is_lucky_loser")):
        df[f"d_{name}"] = df[f"{col}_a"].astype(float) - df[f"{col}_b"].astype(float)
    df["d_prev_retired"] = df["prev_retired_a"].astype(float) - df["prev_retired_b"].astype(float)
    df["d_f_n_stats"] = df["f_n_stats_a"].astype(float) - df["f_n_stats_b"].astype(float)
    ra, rb = df["rank_a"].astype(float), df["rank_b"].astype(float)
    ok = (ra > 0) & (rb > 0)
    df["log_rank_sum"] = np.where(ok, np.log(ra.where(ok)) + np.log(rb.where(ok)), np.nan)

    surf = df["surface"].replace({"Carpet": "Hard"})
    for k, name in (("Hard", "surf_hard"), ("Clay", "surf_clay"), ("Grass", "surf_grass")):
        df[name] = (surf == k).astype(float)
    for lv in LEVELS:
        df[f"lvl_{lv}"] = (df["tourney_level"] == lv).astype(float)
    df["round_ord"] = df["round"].map(ROUNDS).astype(float)
    df["best_of"] = df["best_of"].astype(float)
    df["draw_size"] = df["draw_size"].astype(float)
    df["indoor"] = df["indoor_filled"].map({"I": 1.0, "O": 0.0})
    for tag, sfx in (("s", "static"), ("r", "running")):
        pc = df[f"pace_{sfx}"]
        df[f"d_x_pace_{tag}_ace"] = pc * df["d_f_ace_rate"]
        df[f"d_x_pace_{tag}_sv"] = pc * df["d_f_sv_won_pct"]
        df[f"d_x_pace_{tag}_ret"] = pc * df["d_f_ret_won_pct"]
    df["d_x_indoor_ace"] = df["indoor"] * df["d_f_ace_rate"]
    df["d_x_indoor_sv"] = df["indoor"] * df["d_f_sv_won_pct"]
    df["d_x_alt_ace"] = df["altitude_km"] * df["d_f_ace_rate"]
    df["y"] = df["a_wins"].astype(float)
    if ext:
        df = _add_ext(df, aw)
    return df


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def point_model_prob(df: pd.DataFrame, pace_k: float = MARKOV_PACE_K, mix: float = MARKOV_SURFACE_MIX):
    """P(A wins) from the serve/return form of both players, via the tennis scoring system."""
    def blend(overall, surface):
        return np.where(surface.notna(), (1 - mix) * overall + mix * surface, overall)

    sv_a, sv_b = blend(df["sv_a"], df["ssv_a"]), blend(df["sv_b"], df["ssv_b"])
    ret_a, ret_b = blend(df["ret_a"], df["sret_a"]), blend(df["ret_b"], df["sret_b"])
    shift = pace_k * 0.0473 * df["pace_static"].fillna(0.0).to_numpy()
    pa = np.clip(sv_a + (1 - ret_b) - df["mean_sv"].to_numpy() + shift, 0.40, 0.85)
    pb = np.clip(sv_b + (1 - ret_a) - df["mean_sv"].to_numpy() + shift, 0.40, 0.85)
    ok = ~(np.isnan(pa) | np.isnan(pb))
    out = np.full(len(df), np.nan)
    if ok.any():
        out[ok] = pointmodel.win_prob(pa[ok], pb[ok], df["best_of"].to_numpy()[ok])
    return out


def _add_ext(df: pd.DataFrame, aw) -> pd.DataFrame:
    for h in (90, 365):
        w, l = df[f"w_tr{h}"].to_numpy(float), df[f"l_tr{h}"].to_numpy(float)
        df[f"d_elo_tr{h}"] = np.where(aw, w, l) - np.where(aw, l, w)
    gw, gl = df["g_w_g"].to_numpy(float), df["g_l_g"].to_numpy(float)
    rw, rl = df["g_w_rd"].to_numpy(float), df["g_l_rd"].to_numpy(float)
    ga, gb = np.where(aw, gw, gl), np.where(aw, gl, gw)
    ra, rb = np.where(aw, rw, rl), np.where(aw, rl, rw)
    df["d_glicko"] = ga - gb
    df["glicko_sd"] = np.hypot(ra, rb)
    df["d_glicko_p"] = _logit(np.where(aw, df["g_p_w"], 1 - df["g_p_w"]))
    df["d_markov"] = _logit(point_model_prob(df))
    return df


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """Swap the two players: ANTI columns change sign, SYM columns are untouched."""
    M = X.copy()
    for c in M.columns:
        if c not in SYM:
            M[c] = -M[c]
    return M


def split(df: pd.DataFrame, train_end: int, eval_year: int | None = None, eval_from: int | None = None):
    """Train on years <= train_end; evaluate on one year, or on every year >= eval_from."""
    tr = df[df["yr"] <= train_end]
    if eval_year is not None:
        ev = df[df["yr"] == eval_year]
    else:
        ev = df[df["yr"] >= eval_from]
    assert tr["yr"].max() < ev["yr"].min(), "training years must precede evaluation years"
    return tr, ev


def assert_no_test_years(df: pd.DataFrame) -> None:
    if (df["yr"] >= TEST_START_YEAR).any():
        raise RuntimeError("frozen test years leaked into a development frame")

V05_GROUPS = FULL_GROUPS  # features of the shipped v0.5 model; load with ext=True
