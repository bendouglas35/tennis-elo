"""Serve/return point model: from the two players' serve-point win probabilities to a match
probability, using the tennis scoring system (games, tie-breaks, sets, best-of-3/5).

pa = P(A wins a point on A's serve), pb = P(B wins a point on B's serve), independent points.
The first server of each set is unknown before the match, so a set is averaged over both orders,
which makes win_prob(pa, pb) + win_prob(pb, pa) = 1 exactly.
"""
from __future__ import annotations

import numpy as np


def hold(p):
    """P(server wins a game) from point probability p (deuce solved in closed form)."""
    p = np.asarray(p, float)
    q = 1.0 - p
    return p ** 4 * (1 + 4 * q + 10 * q ** 2) + 20 * p ** 3 * q ** 3 * p ** 2 / (1 - 2 * p * q)


def tiebreak(pf, ps):
    """P(first server wins the tie-break); pf = first server's serve-point prob, ps = the other's."""
    pf, ps = np.asarray(pf, float), np.asarray(ps, float)
    w, l = pf * (1 - ps), (1 - pf) * ps                    # win / lose both points of a pair
    tie = w / (w + l)
    # who serves point t: 0 = first server, pattern 0,1,1,0,0,1,1,0,...
    win_pt = lambda t: pf if ((t + 1) // 2) % 2 == 0 else 1 - ps   # P(first server wins point t)
    V = {}
    for i in range(6, -1, -1):
        for j in range(6, -1, -1):
            if (i, j) == (6, 6):
                V[i, j] = tie
                continue
            pt = win_pt(i + j)
            up = 1.0 if i + 1 >= 7 and (i + 1 - j) >= 2 else V.get((i + 1, j))
            dn = 0.0 if j + 1 >= 7 and (j + 1 - i) >= 2 else V.get((i, j + 1))
            V[i, j] = pt * up + (1 - pt) * dn
    return V[0, 0]


def _set_first(pa, pb):
    """P(A wins a set) when A serves first. Games alternate servers; 6-6 goes to a tie-break."""
    ha, hb = hold(pa), hold(pb)
    tb = tiebreak(pa, pb)                                  # game 12 is served by the first server
    V = {}
    for t in range(12, -1, -1):
        for ga in range(min(t, 7), -1, -1):
            gb = t - ga
            if gb > 7 or gb < 0:
                continue
            if (ga, gb) == (6, 6):
                V[ga, gb] = tb
                continue
            win_game = ha if t % 2 == 0 else 1 - hb        # P(A wins game t)
            def nxt(a, b):
                if a >= 6 and a - b >= 2:
                    return 1.0
                if b >= 6 and b - a >= 2:
                    return 0.0
                if a == 7:
                    return 1.0
                if b == 7:
                    return 0.0
                return V[a, b]
            V[ga, gb] = win_game * nxt(ga + 1, gb) + (1 - win_game) * nxt(ga, gb + 1)
    return V[0, 0]


def set_prob(pa, pb):
    return 0.5 * (_set_first(pa, pb) + 1.0 - _set_first(pb, pa))


def win_prob(pa, pb, best_of=3):
    """P(A beats B). pa, pb arrays; best_of scalar or array of 3 / 5."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    s = set_prob(pa, pb)
    p3 = s * s * (3 - 2 * s)
    p5 = s ** 3 * (10 - 15 * s + 6 * s * s)
    return np.where(np.asarray(best_of) == 5, p5, p3)
