import pytest

from tennis.score import parse_score


def test_straight_sets_with_tiebreak():
    p = parse_score("6-3 7-6(4) 6-2")
    assert p.status == "completed"
    assert (p.sets_w, p.sets_l) == (3, 0)
    assert (p.games_w, p.games_l) == (19, 11)
    assert p.tiebreaks == 1 and not p.last_set_partial


def test_five_setter():
    p = parse_score("6-4 3-6 7-6(5) 3-6 6-4")
    assert p.status == "completed"
    assert (p.sets_w, p.sets_l) == (3, 2)
    assert p.tiebreaks == 1


def test_retirement_mid_set():
    p = parse_score("6-3 2-0 RET")
    assert p.status == "retired"
    assert (p.sets_w, p.sets_l) == (1, 0)
    assert (p.games_w, p.games_l) == (8, 3)
    assert p.last_set_partial


def test_retirement_after_completed_sets_is_not_partial():
    p = parse_score("6-4 6-4 ret")
    assert p.status == "retired" and not p.last_set_partial


def test_walkover_and_default():
    assert parse_score("W/O").status == "walkover"
    d = parse_score("6-3 DEF")
    assert d.status == "default" and d.sets_w == 1


def test_match_tiebreak_not_counted_as_games():
    p = parse_score("7-6(4) 6-7(5) [10-8]")
    assert p.status == "completed" and p.match_tiebreak
    assert (p.sets_w, p.sets_l) == (2, 1)
    assert (p.games_w, p.games_l) == (13, 13)
    assert p.tiebreaks == 2


def test_long_final_set():
    p = parse_score("13-11 6-4")
    assert (p.sets_w, p.games_w, p.games_l) == (2, 19, 15)


@pytest.mark.parametrize("bad", [None, "", "   ", "garbage", "6-3 x"])
def test_unparseable_is_unknown(bad):
    assert parse_score(bad).status == "unknown"
