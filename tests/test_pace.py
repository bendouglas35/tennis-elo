"""Court pace: hand-computed residuals, pooling and shrinkage, and mutation tests of the leakage
checks against deliberately broken pace SQL."""
import datetime as dt

import duckdb
import pandas as pd
import pytest

from tennis import leakage, pace
from tennis.build import build
from tennis.pace import PaceParams


def _con():
    con = duckdb.connect()
    con.execute("""create table matches (match_id varchar, tourney_id varchar, event_id varchar,
        surface varchar, indoor_filled varchar, tourney_date date, ord_num bigint, is_rateable boolean,
        is_team_event boolean, is_nonstandard_format boolean)""")
    return con


def _add_match(con, mid, tid, event, date, ord_num=0, surface="Hard", ind="O", team=False):
    con.execute("insert into matches values (?,?,?,?,?,?,?,true,?,false)",
                [mid, tid, event, surface, ind, date, ord_num, team])


def test_match_residuals_hand_computed():
    con = _con()
    con.execute("""create table player_matches (match_id varchar, tourney_id varchar, player_id varchar, tourney_date date,
        won boolean, ace int, svpt int, sv_pts_won int, has_stats boolean, is_team_event boolean)""")
    con.execute("""create table form_overall (match_id varchar, player_id varchar, f_ace_rate double,
        f_ace_allowed_rate double, f_sv_won_pct double, f_ret_won_pct double)""")
    d0, d1 = dt.date(2022, 1, 3), dt.date(2022, 1, 10)
    # E0 (earlier week) only sets the tour means: aces 16/200 = 0.08, serve points won 130/200 = 0.65.
    _add_match(con, "E0", "T0", "e0", d0)
    _add_match(con, "M1", "T1", "e1", d1)
    _add_match(con, "TEAM", "T2", "e2", d1, team=True)
    pm = [("E0", "T0", "a", d0, True, 10, 100, 60), ("E0", "T0", "b", d0, False, 6, 100, 70),
          ("M1", "T1", "a", d1, True, 8, 80, 56), ("M1", "T1", "b", d1, False, 6, 120, 78),
          ("TEAM", "T2", "c", d1, True, 8, 80, 56), ("TEAM", "T2", "d", d1, False, 6, 120, 78)]
    con.executemany("insert into player_matches values (?,?,?,?,?,?,?,?,true,false)", pm)
    forms = {"a": (0.10, 0.06, 0.70, 0.40), "b": (0.05, 0.08, 0.60, 0.35)}
    for mid in ("E0", "M1", "TEAM"):
        for pid, f in forms.items():
            con.execute("insert into form_overall values (?,?,?,?,?,?)",
                        [mid, pid if mid != "TEAM" else {"a": "c", "b": "d"}[pid], *f])
    con.execute(pace.tour_mean_sql())
    con.execute(pace.match_sql(sd_ace=0.01, sd_sp=0.1))
    rows = con.execute("select match_id, ace_res, sp_res, comp from pace_match").fetchall()
    # E0 has no earlier data for the tour mean, TEAM is a team event: only M1 survives.
    assert [r[0] for r in rows] == ["M1"]
    # expected aces = 80*.10*.08/.08 + 120*.05*.06/.08 = 12.5 vs 14 actual -> 1.5 / 200 serve points
    # expected points won = 80*(.70+1-.35-.65) + 120*(.60+1-.40-.65) = 122 vs 134 -> 12 / 200
    assert rows[0][1] == pytest.approx(1.5 / 200)
    assert rows[0][2] == pytest.approx(12 / 200)
    assert rows[0][3] == pytest.approx(0.5 * (0.0075 / 0.01 + 0.06 / 0.1))


def test_static_and_running_pace_hand_computed():
    con = _con()
    con.execute("create table pace_match (match_id varchar, tourney_id varchar, yr int, "
                "ace_res double, sp_res double, comp double)")
    comps = []

    def edition(tid, event, year, n, total, ind="O"):
        for i in range(n):
            mid = f"{tid}-{i}"
            _add_match(con, mid, tid, event, dt.date(year, 6, 1), ind=ind)
            comps.append((mid, tid, year, 0.0, 0.0, total / n))

    edition("X19", "X", 2019, 10, 5.0)
    edition("X20", "X", 2020, 20, -4.0)
    edition("X21", "X", 2021, 40, 16.0)
    edition("X21i", "X", 2021, 10, 100.0, ind="I")      # same event, indoor: must not be pooled
    edition("Y19", "Y", 2019, 30, -3.0)
    edition("X22b", "X", 2022, 5, 50.0)                  # earlier edition in the SAME year: not pooled
    con.execute("update matches set tourney_date = date '2022-03-01' where tourney_id = 'X22b'")
    for mid, ordn, comp in (("A1", 1000, 2.0), ("A2", 1000, 4.0), ("A3", 2000, None), ("A4", 3000, 0.0)):
        _add_match(con, mid, "X22", "X", dt.date(2022, 6, 1), ord_num=ordn)
        if comp is not None:
            comps.append((mid, "X22", 2022, 0.0, 0.0, comp))
    _add_match(con, "MAD", "M22", "1536", dt.date(2022, 5, 1))   # Madrid: altitude lookup
    con.register("c", pd.DataFrame(comps, columns=["match_id", "tourney_id", "yr", "ace_res", "sp_res", "comp"]))
    con.execute("insert into pace_match select * from c")

    params = PaceParams(rho=0.5, editions=2, m_static=10.0, m_running=4.0)
    con.execute(pace.edition_sql())
    con.execute(pace.static_sql(params))
    con.execute(pace.altitude_sql())
    con.execute(pace.final_sql(params))

    mu0 = (5.0 - 4.0 + 16.0 - 3.0) / (10 + 20 + 40 + 30)         # earlier Hard/outdoor editions
    static = (16.0 * 1 + (-4.0) * 0.5 + 10 * mu0) / (40 * 1 + 20 * 0.5 + 10)  # 2021 age 1, 2020 age 2
    got = {r[0]: r[1:] for r in con.execute(
        "select match_id, pace_static, pace_running, pace_n_running, pace_n_prior, altitude_km "
        "from match_pace where match_id in ('A1','A2','A3','A4','MAD','X19-0')").fetchall()}
    running = (2.0 + 4.0 + 4.0 * static) / (2 + 4.0)
    assert got["A1"] == pytest.approx((static, static, 0, 2, 0.0))     # simultaneous: no running info
    assert got["A2"] == pytest.approx((static, static, 0, 2, 0.0))
    assert got["A3"] == pytest.approx((static, running, 2, 2, 0.0))
    assert got["A4"] == pytest.approx((static, running, 2, 2, 0.0))    # A3 has no composite: not counted
    assert got["X19-0"][0] == pytest.approx(0.0)                       # first edition: prior mean 0
    assert got["X19-0"][3] == 0
    assert got["MAD"][4] == pytest.approx(0.65)


def _rich_db(tmp_path, rich_raw_dir):
    db = tmp_path / "rich.duckdb"
    build(db, rich_raw_dir)
    return str(db)


def test_pace_is_present_and_event_level_on_pipeline_data(tmp_path, rich_raw_dir):
    con = duckdb.connect(_rich_db(tmp_path, rich_raw_dir), read_only=True)
    n, null_static, valid = con.execute(
        "select count(*), sum((pace_static is null)::int), count(pace_running) from model_features").fetchone()
    assert n > 0 and null_static == 0 and valid == n
    # event-level: every match of one tournament shares the same static pace
    assert con.execute("select max(c) from (select count(distinct pace_static) c from model_features "
                       "group by tourney_id)").fetchone()[0] == 1


@pytest.mark.parametrize("target, old, new", [
    ("static_sql", "p.yr < e.yr", "p.yr <= e.yr"),                 # pool the edition's own year
    ("final_sql", "and 1 preceding", "and current row"),           # running pace sees its own match
])
def test_leakage_check_catches_broken_pace(tmp_path, rich_raw_dir, monkeypatch, target, old, new):
    db = _rich_db(tmp_path, rich_raw_dir)
    original = getattr(pace, target)
    broken = lambda *a, **k: (lambda sql: (_ for _ in ()).throw(AssertionError("mutation did not apply"))
                              if old not in sql else sql.replace(old, new))(original(*a, **k))
    monkeypatch.setattr(pace, target, broken)
    r = leakage.perturbation_check(db, n=10, rounds=6, min_sample=1)
    assert not r["ok"] and r["own_rows_changed"] > 0, r
