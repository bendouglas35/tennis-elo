import math

import numpy as np
import pandas as pd
import pytest

from tennis.glicko import BASE, Q, RD0, GlickoConfig, _g, run_glicko


def _df(rows):
    recs = [dict(match_id=f"m{i}", ord_num=o, day=d, winner_id=w, loser_id=l, standard=True, aux=False)
            for i, (o, d, w, l) in enumerate(rows)]
    return pd.DataFrame(recs)


def test_first_match_update_matches_the_glicko_formula():
    out = run_glicko(_df([(1, 0, "A", "B"), (2, 7, "A", "B")]), GlickoConfig(c=0.0, rd_min=30))
    assert out.loc[0, ["w_g", "w_rd", "l_rd"]].tolist() == [BASE, RD0, RD0]
    g, e = _g(RD0), 0.5
    d2 = 1 / (Q * Q * g * g * e * (1 - e))
    prec = 1 / RD0 ** 2 + 1 / d2
    assert out.loc[1, "w_g"] == pytest.approx(BASE + Q / prec * g * 0.5)
    assert out.loc[1, "l_g"] == pytest.approx(BASE - Q / prec * g * 0.5)
    assert out.loc[1, "w_rd"] == pytest.approx(math.sqrt(1 / prec))


def test_deviation_shrinks_with_matches_and_grows_with_inactivity():
    rows = [(i, 7 * i, "A", f"X{i}") for i in range(1, 12)] + [(99, 7 * 11 + 400, "A", "Z")]
    out = run_glicko(_df(rows), GlickoConfig(c=35.0, rd_min=30))
    assert out["w_rd"].iloc[1] < out["w_rd"].iloc[0]
    assert out["w_rd"].iloc[10] < out["w_rd"].iloc[1]
    assert out["w_rd"].iloc[-1] > out["w_rd"].iloc[10] * 1.15       # a 400-day layoff restores doubt


def test_simultaneous_matches_and_future_independence():
    df = _df([(1, 0, "A", "B"), (2, 7, "A", "C"), (2, 7, "D", "A"), (3, 14, "A", "B")])
    out = run_glicko(df, GlickoConfig(c=0))
    assert out.loc[1, "w_g"] == out.loc[2, "l_g"]
    cut = run_glicko(df[df["ord_num"] <= 2].reset_index(drop=True), GlickoConfig(c=0))
    pd.testing.assert_frame_equal(out.iloc[: len(cut)], cut)


def test_auxiliary_matches_update_but_are_not_scored():
    df = _df([(1, 0, "A", "B"), (2, 7, "A", "C")])
    df["aux"] = [True, False]
    df["standard"] = [False, True]
    assert run_glicko(df, GlickoConfig(use_aux=False)).loc[0, "w_g"] == BASE
    on = run_glicko(df, GlickoConfig(use_aux=True))
    assert len(on) == 1 and on.loc[0, "w_g"] > BASE and on.loc[0, "w_rd"] < RD0


def test_win_probability_is_pulled_towards_half_by_uncertainty():
    out = run_glicko(_df([(i, 7 * i, "A", f"X{i}") for i in range(1, 15)] + [(50, 120, "A", "NEW")]),
                     GlickoConfig(c=0))
    last = out.iloc[-1]
    plain = 1 / (1 + 10 ** (-(last.w_g - last.l_g) / 400))
    assert 0.5 < last.p_w < plain
