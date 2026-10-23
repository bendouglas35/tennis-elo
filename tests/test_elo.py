import math

import numpy as np
import pandas as pd
import pytest

from tennis.elo import EloConfig, bo3_to_bo5, run_elo


def _df(rows):
    """rows: (ord, day, winner, loser, extra dict)"""
    recs = []
    for i, (o, day, w, l, extra) in enumerate(rows):
        r = dict(match_id=f"m{i}", tourney_id=f"t{o}", ord_num=o, day=day, surface="Hard", best_of=3,
                 winner_id=w, loser_id=l, games_w=12, games_l=6, retired=False, standard=True)
        r.update(extra)
        recs.append(r)
    return pd.DataFrame(recs)


def _p(diff):
    return 1 / (1 + 10 ** (-diff / 400))


FIXED = EloConfig(k_mode="fixed", k_fixed=32.0)


def test_fixed_k_updates_match_hand_computation():
    out = run_elo(_df([(1, 0, "A", "B", {}), (2, 7, "A", "B", {})]), FIXED)
    assert out.loc[0, ["w_r", "l_r", "p_w"]].tolist() == pytest.approx([1500, 1500, 0.5])
    a, b = 1500 + 32 * 0.5, 1500 - 32 * 0.5
    assert out.loc[1, ["w_r", "l_r"]].tolist() == pytest.approx([a, b])
    assert out.loc[1, "p_w"] == pytest.approx(_p(a - b))
    assert math.isnan(out.loc[0, "w_gap"]) and out.loc[1, "w_gap"] == 7


def test_dynamic_k_shrinks_with_experience():
    cfg = EloConfig(k0=250, k_off=5, k_shape=0.4)
    out = run_elo(_df([(1, 0, "A", "B", {})]), cfg)
    assert out.loc[0, "w_n"] == 0
    out = run_elo(_df([(1, 0, "A", "B", {}), (2, 7, "A", "B", {})]), cfg)
    k0 = 250 / 5 ** 0.4
    assert out.loc[1, "w_r"] == pytest.approx(1500 + k0 * 0.5)
    assert out.loc[1, "w_n"] == 1


def test_simultaneous_matches_never_feed_each_other():
    rows = [(1, 0, "A", "B", {}), (2, 7, "A", "C", {}), (2, 7, "D", "A", {}), (3, 14, "A", "B", {})]
    out = run_elo(_df(rows), FIXED)
    # both ord=2 matches see A's rating from after match 0 only
    assert out.loc[1, "w_r"] == out.loc[2, "l_r"] == pytest.approx(1516)


def test_team_events_are_not_scored_and_do_not_update_but_count_as_activity():
    rows = [(1, 0, "A", "B", {}), (2, 30, "A", "C", {"standard": False}), (3, 60, "A", "D", {})]
    out = run_elo(_df(rows), FIXED)
    assert len(out) == 2                                   # team match not in the output
    assert out.loc[1, "w_r"] == pytest.approx(1516)        # ... and made no update to A
    assert out.loc[1, "w_gap"] == 30                       # but A was active 30 days earlier


def test_layoff_decays_effective_matches():
    rows = [(1, 0, "A", "B", {}), (2, 400, "A", "B", {})]
    cfg = EloConfig(layoff_grace=100, layoff_tau=200)
    out = run_elo(_df(rows), cfg)
    assert out.loc[1, "w_n"] == pytest.approx(1 * math.exp(-300 / 200))
    assert run_elo(_df(rows), EloConfig()).loc[1, "w_n"] == 1


def test_surface_blend_starts_from_overall_and_is_separate_per_surface():
    rows = [(1, 0, "A", "B", {"surface": "Clay"}), (2, 7, "A", "B", {"surface": "Hard"}),
            (3, 14, "A", "B", {"surface": "Clay"})]
    cfg = EloConfig(k_mode="fixed", k_fixed=32, surface_w=0.5)
    out = run_elo(_df(rows), cfg)
    assert out.loc[0, "w_s"] == 1500                       # nothing yet
    assert out.loc[1, "w_s"] == pytest.approx(1516)        # first hard match starts at overall (1516)
    clay_a = 1500 + 32 * 0.5                               # clay rating after match 0
    assert out.loc[2, "w_s"] == pytest.approx(clay_a)      # untouched by the hard match
    assert out.loc[2, "w_b"] == pytest.approx(0.5 * out.loc[2, "w_r"] + 0.5 * clay_a)


def test_margin_and_retirement_scale_the_update():
    rows = [(1, 0, "A", "B", {"games_w": 12, "games_l": 0}), (2, 7, "C", "D", {"retired": True})]
    out = run_elo(_df(rows), EloConfig(k_mode="fixed", k_fixed=32, mov=1.0, retire_scale=0.5))
    nxt = _df([(1, 0, "A", "B", {"games_w": 12, "games_l": 0}), (2, 7, "A", "B", {})])
    nxt_out = run_elo(nxt, EloConfig(k_mode="fixed", k_fixed=32, mov=1.0))
    assert nxt_out.loc[1, "w_r"] == pytest.approx(1500 + 32 * 0.5 * 2.0)   # shutout doubles the update
    ret = run_elo(_df([(1, 0, "A", "B", {"retired": True}), (2, 7, "A", "B", {})]),
                  EloConfig(k_mode="fixed", k_fixed=32, retire_scale=0.5))
    assert ret.loc[1, "w_r"] == pytest.approx(1500 + 32 * 0.5 * 0.5)
    ign = run_elo(_df([(1, 0, "A", "B", {"retired": True}), (2, 7, "A", "B", {})]),
                  EloConfig(k_mode="fixed", k_fixed=32, retire_scale=0.0))
    assert ign.loc[1, "w_r"] == 1500 and ign.loc[1, "w_n"] == 0


def test_bo5_conversion_is_symmetric_and_sharpens():
    assert bo3_to_bo5(0.5) == pytest.approx(0.5, abs=1e-9)
    assert bo3_to_bo5(0.7) > 0.7
    assert bo3_to_bo5(0.3) == pytest.approx(1 - bo3_to_bo5(0.7), abs=1e-6)
    rows = [(1, 0, "A", "B", {}), (2, 7, "A", "B", {"best_of": 5})]
    plain = run_elo(_df(rows), FIXED).loc[1, "p_w"]
    conv = run_elo(_df(rows), EloConfig(k_mode="fixed", k_fixed=32, bo5=True)).loc[1, "p_w"]
    assert conv > plain


def _random_matches(n=300, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a, b = rng.choice(20, 2, replace=False)
        w, l = (a, b) if rng.random() < 0.6 else (b, a)
        rows.append((i // 3, 7 * (i // 3), f"P{w}", f"P{l}", {"surface": str(rng.choice(["Hard", "Clay"]))}))
    return _df(rows)


FULL = EloConfig(surface_w=0.5, layoff_grace=60, mov=0.5, retire_scale=0.5, bo5=True)


def test_ratings_never_depend_on_the_future():
    df = _random_matches()
    full = run_elo(df, FULL)
    cut = df["ord_num"].quantile(0.6)
    trunc = run_elo(df[df["ord_num"] <= cut].reset_index(drop=True), FULL)
    pd.testing.assert_frame_equal(full.iloc[: len(trunc)], trunc)


def test_a_matchs_own_result_does_not_affect_its_own_pre_match_ratings():
    df = _random_matches()
    base = run_elo(df, FULL)
    i = 200
    flipped = df.copy()
    flipped.loc[i, ["winner_id", "loser_id"]] = df.loc[i, ["loser_id", "winner_id"]].to_numpy()
    flipped.loc[i, "retired"] = not df.loc[i, "retired"]
    alt = run_elo(flipped, FULL)
    assert base.loc[i, "w_r"] == alt.loc[i, "l_r"] and base.loc[i, "l_r"] == alt.loc[i, "w_r"]
    assert (base.loc[:i - 1].drop(columns="match_id").to_numpy() ==
            alt.loc[:i - 1].drop(columns="match_id").to_numpy())[~np.isnan(base.loc[:i - 1].drop(columns="match_id").to_numpy())].all()
    assert not np.allclose(base.loc[i + 1:, "w_r"], alt.loc[i + 1:, "w_r"])    # later rows react


def test_k_floor_and_layoff_boost():
    rows = [(i, 7 * i, "A", "B", {}) for i in range(1, 40)]
    floor = run_elo(_df(rows), EloConfig(k0=250, k_floor=80.0))
    plain = run_elo(_df(rows), EloConfig(k0=250))
    assert floor["w_r"].iloc[-1] > plain["w_r"].iloc[-1]       # floor of 80 binds once K(N) falls below it
    two = [(1, 0, "A", "B", {}), (2, 400, "A", "B", {})]
    boosted = run_elo(_df(two), EloConfig(k_mode="fixed", k_fixed=32, layoff_grace=100, layoff_tau=200,
                                          layoff_boost=1.0))
    dec = math.exp(-300 / 200)
    nxt = run_elo(_df(two + [(3, 407, "A", "B", {})]),
                  EloConfig(k_mode="fixed", k_fixed=32, layoff_grace=100, layoff_tau=200, layoff_boost=1.0))
    a1 = 1500 + 32 * 0.5
    e = _p(a1 - (1500 - 32 * 0.5))
    assert nxt.loc[2, "w_r"] == pytest.approx(a1 + 32 * (1 + 1.0 * (1 - dec)) * (1 - e))
