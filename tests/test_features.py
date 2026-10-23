import datetime as dt
import random

import duckdb
import pytest

from conftest import COLUMNS, make_row, stat_overrides, write_csv
from tennis import features, leakage
from tennis.build import build


def _toy(tmp_path, rows, name="toy", half_life=3.0, columns=COLUMNS):
    d = tmp_path / name
    d.mkdir()
    write_csv(d / "2022.csv", rows, columns)
    db = tmp_path / f"{name}.duckdb"
    build(db, d, min_year=2000, half_life=half_life, min_eff=0.0)
    return duckdb.connect(str(db), read_only=True)


def _date(offset_days, base=dt.date(2022, 1, 3)):
    return (base + dt.timedelta(days=offset_days)).strftime("%Y%m%d")


def _ew(pairs, half_life):
    """Exponentially weighted ratio of (num, den) pairs, oldest first."""
    d = 0.5 ** (1.0 / half_life)
    k = len(pairs)
    num = sum(n * d ** (k - 1 - i) for i, (n, _) in enumerate(pairs))
    den = sum(m * d ** (k - 1 - i) for i, (_, m) in enumerate(pairs))
    return num / den if den > 0 and k else None


def test_ew_form_matches_hand_computation_including_missing_stats_and_role_swap(tmp_path):
    """Player 1 plays 7 weekly matches: wins, wins, LOSES, wins (no stats), wins, wins, wins."""
    rng = random.Random(5)
    plan = [True, True, False, True, True, True, True]
    no_stats = {3}
    rows, stats = [], []
    for i, wins in enumerate(plan):
        ov = stat_overrides(rng)
        if i in no_stats:
            ov.update({k: "" for k in ov if k != "minutes"})
        w, l = (1, 100 + i) if wins else (100 + i, 1)
        rows.append(make_row(2022, 1, w, l, wrank=w, lrank=l, tourney_id=f"2022-{i}",
                             tourney_date=_date(7 * i), match_num=1, **ov))
        stats.append(ov)
    con = _toy(tmp_path, rows, half_life=3.0)

    got = con.execute("""
        select pf.f_sv_won_pct, pf.f_ret_won_pct, pf.f_win_rate
        from player_features pf join matches m using (match_id)
        where pf.player_id = '1' order by m.ord_num""").fetchall()
    assert len(got) == len(plan)

    for m in range(len(plan)):
        prior = [j for j in range(m) if j not in no_stats]
        sv = _ew([(stats[j][f"{'w' if plan[j] else 'l'}_1stWon"] + stats[j][f"{'w' if plan[j] else 'l'}_2ndWon"],
                   stats[j][f"{'w' if plan[j] else 'l'}_svpt"]) for j in prior], 3.0)
        ret = _ew([(stats[j][f"{'l' if plan[j] else 'w'}_svpt"] - stats[j][f"{'l' if plan[j] else 'w'}_1stWon"]
                    - stats[j][f"{'l' if plan[j] else 'w'}_2ndWon"],
                    stats[j][f"{'l' if plan[j] else 'w'}_svpt"]) for j in prior], 3.0)
        win = _ew([(int(plan[j]), 1) for j in range(m)], 3.0)  # every match counts, incl. no-stats
        for value, expected in zip(got[m], (sv, ret, win)):
            if expected is None:
                assert value is None
            else:
                assert value == pytest.approx(expected, rel=1e-9)


def test_fatigue_and_rest_features(tmp_path):
    rows = []
    # Tournament A (week 0): player 1 wins R32, R16, QF.  B (week 1): loses R32 by retirement.
    # C (week 3): R32.
    spec = [("2022-A", 0, "R32", 1, 11, "6-3 6-4", 100), ("2022-A", 0, "R16", 1, 12, "7-6(4) 6-3", 120),
            ("2022-A", 0, "QF", 1, 13, "6-4 6-4", 90), ("2022-B", 7, "R32", 14, 1, "6-3 2-0 RET", 40),
            ("2022-C", 21, "R32", 1, 15, "6-3 6-4", 80)]
    for i, (tid, off, rnd, w, l, score, mins) in enumerate(spec):
        rows.append(make_row(2022, 1, w, l, score=score, tourney_id=tid, tourney_date=_date(off),
                             round=rnd, match_num=i + 1, minutes=mins))
    con = _toy(tmp_path, rows)
    cols = ("tourney_matches_before, tourney_minutes_before, tourney_sets_before, days_since_last, "
            "n_14d, n_28d, career_matches, prev_retired")
    got = con.execute(f"""
        select {cols} from fatigue fa join matches m using (match_id)
        where fa.player_id = '1' order by m.ord_num""").fetchall()
    assert got == [
        (0, 0, 0, None, 0, 0, 0, None),   # A R32: first match ever
        (1, 100, 2, 0, 1, 1, 1, 0),       # A R16
        (2, 220, 4, 0, 2, 2, 2, 0),       # A QF: 2 rounds, 220 minutes, 4 sets before
        (0, 0, 0, 7, 3, 3, 3, 0),         # B R32 a week later: A's 3 matches are inside 14 days
        (0, 0, 0, 14, 0, 4, 4, 1),        # C: B is exactly 14 days back (outside 14d), A 21 (inside 28d); B ended in RET
    ]


@pytest.mark.parametrize("first_round, second_round, second_offset, expected", [
    ("R128", "F", 13, 1),   # 13 days apart, worst case for inclusion (round ranks 1 then 7)
    ("F", "R128", 14, 0),   # 14 days apart, worst case for exclusion (round ranks 7 then 1)
])
def test_14_day_window_boundary_is_exact(tmp_path, first_round, second_round, second_offset, expected):
    rows = [make_row(2022, 1, 1, 21, tourney_id="2022-A", tourney_date=_date(0), round=first_round, match_num=1),
            make_row(2022, 1, 1, 22, tourney_id="2022-B", tourney_date=_date(second_offset),
                     round=second_round, match_num=1)]
    con = _toy(tmp_path, rows)
    n14 = con.execute("""select n_14d from fatigue fa join matches m using (match_id)
                         where fa.player_id = '1' order by m.ord_num""").fetchall()
    assert n14 == [(0,), (expected,)]


def test_round_robin_uses_match_num_only_when_present(tmp_path):
    def rr_rows(m1, m2):
        return [make_row(2022, 1, 1, 21, tourney_id="2022-F", round="RR", match_num=m1),
                make_row(2022, 1, 1, 22, tourney_id="2022-F", round="RR", match_num=m2)]

    ordered = _toy(tmp_path, rr_rows(5, 9), name="rr_a")
    assert ordered.execute("""select tourney_matches_before from fatigue fa join matches m using (match_id)
                              where fa.player_id = '1' order by m.ord_num""").fetchall() == [(0,), (1,)]
    simultaneous = _toy(tmp_path, rr_rows("", ""), name="rr_b")  # no match_num: equal key, no leakage
    assert simultaneous.execute("select tourney_matches_before from fatigue where player_id = '1'").fetchall() \
        == [(0,), (0,)]


def test_entry_flags_and_indoor_fill(tmp_path):
    cols = COLUMNS + ["indoor"]
    rows = [make_row(2020, 1, 1, 2, tourney_id="2020-900", indoor="I", winner_entry="Q", winner_seed="3"),
            make_row(2021, 1, 1, 2, tourney_id="2021-900", indoor="", loser_entry="WC"),
            make_row(2020, 1, 3, 4, tourney_id="2020-901", indoor="I"),
            make_row(2021, 1, 3, 4, tourney_id="2021-901", indoor="O"),
            make_row(2022, 1, 3, 4, tourney_id="2022-901", indoor="")]
    d = tmp_path / "ind"
    d.mkdir()
    for year in (2020, 2021, 2022):
        write_csv(d / f"{year}.csv", [r for r in rows if r["tourney_date"].startswith(str(year))], cols)
    db = tmp_path / "ind.duckdb"
    build(db, d, min_year=2000)
    con = duckdb.connect(str(db), read_only=True)
    filled = dict(con.execute("select tourney_id, indoor_filled from matches").fetchall())
    assert filled["2021-900"] == "I"          # unambiguous event: filled from other years
    assert filled["2022-901"] is None         # event seen as both I and O: left NULL
    pf = con.execute("""select pf.is_qualifier, pf.is_seeded, pf.is_wildcard from player_features pf
                        where match_id = '2020-900-1' and player_id = '1'""").fetchone()
    assert pf == (True, True, False)


def test_model_features_row_per_side_and_antisymmetric_diffs(tmp_path, rich_raw_dir):
    db = tmp_path / "rich.duckdb"
    build(db, rich_raw_dir)
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("select count(*) from model_features").fetchone() == \
        con.execute("select count(*) from match_sides").fetchone()
    # d_* must equal A minus B of the underlying player features
    bad = con.execute("""
        select count(*) from model_features mf
        join player_features pa on pa.match_id = mf.match_id and pa.player_id = mf.player_a_id
        join player_features pb on pb.match_id = mf.match_id and pb.player_id = mf.player_b_id
        where abs(coalesce(mf.d_f_sv_won_pct, 0) - coalesce(pa.f_sv_won_pct - pb.f_sv_won_pct, 0)) > 1e-12""").fetchone()[0]
    assert bad == 0


def test_leakage_truncation_and_perturbation_pass(tmp_path, rich_raw_dir):
    db = tmp_path / "rich.duckdb"
    build(db, rich_raw_dir)
    t = leakage.truncation_check(str(db))
    assert t["ok"], t
    p = leakage.perturbation_check(str(db), n=10, rounds=6, min_sample=1)
    assert p["ok"], p
    assert p["perturbed"] >= 5 and p["later_rows_changed"] > 0  # not vacuous


def test_leakage_check_catches_an_injected_leak(tmp_path, rich_raw_dir, monkeypatch):
    db = tmp_path / "rich.duckdb"
    build(db, rich_raw_dir)
    monkeypatch.setattr(features, "_UNBOUNDED", "range between unbounded preceding and current row")
    p = leakage.perturbation_check(str(db), n=10, rounds=6, min_sample=1)
    assert not p["ok"] and p["own_rows_changed"] > 0


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_prev_retired_is_deterministic_when_latest_matches_share_an_order_key(tmp_path, order):
    """Two matches with an identical ordering key (as in Davis Cup ties), one ending in retirement:
    the next match must see prev_retired = 1 whichever way the rows happen to be stored."""
    tied = [make_row(2022, 1, 1, 11, tourney_id="2022-A", tourney_date=_date(0), round="R32", match_num=1),
            make_row(2022, 1, 12, 1, tourney_id="2022-A", tourney_date=_date(0), round="R32", match_num=2,
                     score="6-3 2-0 RET")]
    rows = [tied[i] for i in order] + [
        make_row(2022, 1, 1, 13, tourney_id="2022-B", tourney_date=_date(7), round="R32", match_num=1)]
    con = _toy(tmp_path, rows)
    got = con.execute("select prev_retired from fatigue where match_id = '2022-B-1' and player_id = '1'")
    assert got.fetchone() == (1,)


def test_indoor_zero_is_normalised_to_outdoor(tmp_path):
    cols = COLUMNS + ["indoor"]
    rows = [make_row(2020, 1, 1, 2, tourney_id="2020-900", indoor="O"),
            make_row(2021, 1, 1, 2, tourney_id="2021-900", indoor="0"),
            make_row(2022, 1, 1, 2, tourney_id="2022-900", indoor="")]
    d = tmp_path / "ind0"
    d.mkdir()
    for year in (2020, 2021, 2022):
        write_csv(d / f"{year}.csv", [r for r in rows if r["tourney_date"].startswith(str(year))], cols)
    db = tmp_path / "ind0.duckdb"
    build(db, d, min_year=2000)
    con = duckdb.connect(str(db), read_only=True)
    got = dict(con.execute("select tourney_id, indoor_filled from matches").fetchall())
    assert got == {"2020-900": "O", "2021-900": "O", "2022-900": "O"}
