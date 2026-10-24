import numpy as np
import pytest

from tennis import pointmodel as pm


def test_game_and_tiebreak_known_values():
    assert float(pm.hold(0.6)) == pytest.approx(0.7357, abs=2e-4)
    assert float(pm.hold(0.5)) == pytest.approx(0.5)
    assert float(pm.tiebreak(0.6, 0.6)) == pytest.approx(0.5, abs=0.05)   # first server slightly ahead
    assert float(pm.tiebreak(0.5, 0.5)) == pytest.approx(0.5)


def test_symmetry_monotonicity_and_best_of_five_sharpens():
    pa, pb = np.array([0.62, 0.70, 0.55]), np.array([0.62, 0.60, 0.66])
    p = pm.win_prob(pa, pb, 3)
    assert p[0] == pytest.approx(0.5)
    assert np.allclose(p + pm.win_prob(pb, pa, 3), 1.0)
    assert (pm.win_prob(pa + 0.02, pb, 3) > p).all()
    assert pm.win_prob(0.66, 0.62, 5) > pm.win_prob(0.66, 0.62, 3) > 0.5


def _simulate(pa, pb, best_of, n, seed):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        sets_a = sets_b = 0
        first = rng.random() < 0.5                       # A serves first in set 1
        need = best_of // 2 + 1
        while sets_a < need and sets_b < need:
            ga = gb = 0
            serve_a = first
            while True:
                if ga == 6 and gb == 6:                  # tie-break
                    pa_, pb_ = 0, 0
                    t, a_srv = 0, serve_a
                    while True:
                        server_a = a_srv if ((t + 1) // 2) % 2 == 0 else not a_srv
                        p_a_wins = pa if server_a else 1 - pb
                        if rng.random() < p_a_wins:
                            pa_ += 1
                        else:
                            pb_ += 1
                        t += 1
                        if (pa_ >= 7 or pb_ >= 7) and abs(pa_ - pb_) >= 2:
                            break
                    ga, gb = (7, 6) if pa_ > pb_ else (6, 7)
                    break
                p_a_game = pa if serve_a else 1 - pb     # point prob for A in this game
                a_pts = b_pts = 0
                while True:
                    if rng.random() < p_a_game:
                        a_pts += 1
                    else:
                        b_pts += 1
                    if (a_pts >= 4 or b_pts >= 4) and abs(a_pts - b_pts) >= 2:
                        break
                if a_pts > b_pts:
                    ga += 1
                else:
                    gb += 1
                serve_a = not serve_a
                if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
                    break
            if ga > gb:
                sets_a += 1
            else:
                sets_b += 1
            first = not first if (ga + gb) % 2 == 0 else first    # next set: continue the alternation
        wins += sets_a > sets_b
    return wins / n


@pytest.mark.parametrize("pa, pb, bo", [(0.66, 0.62, 3), (0.70, 0.58, 5)])
def test_dp_matches_monte_carlo(pa, pb, bo):
    sim = _simulate(pa, pb, bo, 6000, seed=1)
    assert float(pm.win_prob(pa, pb, bo)) == pytest.approx(sim, abs=0.02)
