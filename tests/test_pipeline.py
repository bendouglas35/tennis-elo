import duckdb
import numpy as np

from tennis import baseline
from tennis.build import build
from tennis.download import match_files
from tennis.metrics import accuracy, brier, log_loss


def _build(raw_dir, tmp_path, name="t.duckdb"):
    db = tmp_path / name
    counts = build(db, raw_dir, min_year=2000)
    return db, counts


def test_only_tour_level_files_are_loaded(raw_dir):
    names = [p.name for p in match_files(raw_dir)]
    assert names == ["atp_matches_2020.csv", "atp_matches_2021.csv", "atp_matches_2022.csv"]


def test_tennismylife_layout_with_indoor_column(tmp_path):
    from conftest import COLUMNS, make_row, write_csv

    d = tmp_path / "tml"
    d.mkdir()
    for year in (2021, 2022):
        rows = [make_row(year, n, 1 + n % 50, 60 + n % 40, indoor="I" if n % 2 else "O")
                for n in range(1, 41)]
        write_csv(d / f"{year}.csv", rows, COLUMNS + ["indoor"])
    assert [p.name for p in match_files(d)] == ["2021.csv", "2022.csv"]
    db, c = _build(d, tmp_path, "tml.duckdb")
    assert c["matches"] == 80
    con = duckdb.connect(str(db), read_only=True)
    assert {r[0] for r in con.execute("select distinct indoor from match_sides").fetchall()} == {"I", "O"}


def test_missing_match_num_gets_surrogate_unique_match_id(tmp_path):
    from conftest import make_row, write_csv

    d = tmp_path / "nomatchnum"
    d.mkdir()
    rows = [make_row(2025, 1, 10, 20, match_num="", round="R32"),
            make_row(2025, 2, 11, 21, match_num="", round="R32"),
            make_row(2025, 3, 10, 22, match_num="", round="R16"),
            make_row(2025, 4, 12, 23)]  # normal row alongside
    write_csv(d / "2025.csv", rows)
    db, c = _build(d, tmp_path, "nm.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    assert c["matches"] == 4
    assert con.execute("select count(*) from matches where match_id is null").fetchone()[0] == 0
    assert con.execute("select count(distinct match_id) from matches").fetchone()[0] == 4


def test_team_and_nonstandard_event_flags(tmp_path):
    from conftest import make_row, write_csv

    d = tmp_path / "events"
    d.mkdir()
    rows = [make_row(2024, 1, 1, 2, tourney_name="Laver Cup", tourney_level="A"),
            make_row(2024, 2, 3, 4, tourney_name="United Cup", tourney_level="A"),
            make_row(2024, 3, 5, 6, tourney_name="Atp Cup", tourney_level="A"),
            make_row(2024, 4, 7, 8, tourney_name="Davis Cup G1 R1: A vs B", tourney_level="D"),
            make_row(2024, 5, 9, 10, tourney_name="Next Gen ATP Finals", tourney_level="A"),
            make_row(2024, 6, 11, 12, tourney_name="Brisbane", tourney_level="250")]
    write_csv(d / "2024.csv", rows)
    db, _ = _build(d, tmp_path, "ev.duckdb")
    got = dict(duckdb.connect(str(db), read_only=True).execute(
        "select match_num, is_team_event::int * 10 + is_nonstandard_format::int from matches").fetchall())
    assert got == {1: 10, 2: 10, 3: 10, 4: 10, 5: 1, 6: 0}


def test_sackmann_layout_has_null_indoor(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("select count(indoor) from matches").fetchone()[0] == 0


def test_mixing_sources_in_one_directory_is_refused(raw_dir):
    import pytest
    from conftest import make_row, write_csv

    write_csv(raw_dir / "2021.csv", [make_row(2021, 1, 7, 8)])  # 2021 now exists in both styles
    with pytest.raises(SystemExit, match="appears twice"):
        match_files(raw_dir)


def test_counts_and_rateable_flags(raw_dir, tmp_path):
    db, c = _build(raw_dir, tmp_path)
    assert c["matches"] == 3 * 203
    assert c["rateable"] == 3 * 202            # walkovers excluded
    assert c["match_sides"] == c["rateable"]
    assert c["player_matches"] == 2 * c["rateable"]
    assert c["unknown_scores"] == 0


def test_walkover_not_in_modelling_tables(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("select count(*) from match_sides where match_num = 902").fetchone()[0] == 0
    assert con.execute("select count(*) from player_matches where match_num = 902").fetchone()[0] == 0
    statuses = con.execute(
        "select distinct score_status, is_rateable from matches where match_num = 902").fetchall()
    assert statuses == [("walkover", False)]


def test_sides_label_matches_winner_and_orientation_is_balanced(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    bad = con.execute(
        """
        select count(*) from match_sides s join matches m using (match_id)
        where (s.a_wins and s.player_a_id <> m.winner_id)
           or (not s.a_wins and s.player_a_id <> m.loser_id)
           or s.player_a_id = s.player_b_id
        """
    ).fetchone()[0]
    assert bad == 0
    share = con.execute("select avg(a_wins::int) from match_sides").fetchone()[0]
    assert 0.40 < share < 0.60  # symmetric: not always the winner first


def test_orientation_does_not_depend_on_outcome_or_ranks(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    # Higher-ranked player always wins in the fixture, so if orientation leaked the outcome,
    # rank_a < rank_b would be (almost) always true or always false.
    share = con.execute("select avg((rank_a < rank_b)::int) from match_sides where rank_a is not null").fetchone()[0]
    assert 0.40 < share < 0.60


def test_return_points_are_inverted_opponent_serve_stats(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    row = con.execute(
        "select won, svpt, sv_pts_won, ret_pts_played, ret_pts_won from player_matches "
        "where match_num = 1 and tourney_date >= '2020-01-01' and tourney_date < '2021-01-01' order by won desc"
    ).fetchall()
    winner, loser = row
    # fixture: winner serve 60 pts (30+10 won), loser serve 70 pts (28+12 won)
    assert winner == (True, 60, 40, 70, 70 - 28 - 12)
    assert loser == (False, 70, 40, 60, 60 - 30 - 10)


def test_build_is_deterministic(raw_dir, tmp_path):
    db1, _ = _build(raw_dir, tmp_path, "a.duckdb")
    db2, _ = _build(raw_dir, tmp_path, "b.duckdb")
    q = "select match_id, player_a_id, a_wins from match_sides order by match_id"
    r1 = duckdb.connect(str(db1), read_only=True).execute(q).fetchall()
    r2 = duckdb.connect(str(db2), read_only=True).execute(q).fetchall()
    assert r1 == r2


def test_baseline_runs(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    res = baseline.run(db)
    # fixture: lower rank number always wins -> baseline should be near perfect
    assert res["val_accuracy_higher_ranked"] > 0.99
    assert res["beta"][1] > 0
    assert res["n_val"] > 0 and res["n_train"] > 0


def test_baseline_never_touches_frozen_test_years(raw_dir, tmp_path):
    db, _ = _build(raw_dir, tmp_path)
    before = baseline.run(db)
    con = duckdb.connect(str(db))
    con.execute(
        "insert into match_sides select * replace (date '2023-03-01' as tourney_date) "
        "from match_sides limit 100"
    )
    con.close()
    after = baseline.run(db)
    assert (after["n_train"], after["n_val"]) == (before["n_train"], before["n_val"])
    assert after["val_log_loss"] == before["val_log_loss"]


def test_fit_logistic_recovers_sign():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.3 + 1.5 * x)))).astype(float)
    b = baseline.fit_logistic(x, y)
    assert abs(b[0] - 0.3) < 0.15 and abs(b[1] - 1.5) < 0.15


def test_metrics():
    y = np.array([1, 0, 1, 0])
    assert accuracy(y, np.array([0.9, 0.1, 0.8, 0.4])) == 1.0
    assert abs(log_loss(y, np.full(4, 0.5)) - np.log(2)) < 1e-9
    assert abs(brier(y, np.full(4, 0.5)) - 0.25) < 1e-9
