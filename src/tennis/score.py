"""Parse Sackmann-style score strings.

Scores are always written from the *match winner's* perspective, e.g.
"6-3 7-6(4) 6-2", "6-3 2-0 RET", "W/O", "7-6(4) 6-7(5) [10-8]".
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

_SET_RE = re.compile(r"^(\d+)-(\d+)(?:\((\d+)\))?$")
_MATCH_TB_RE = re.compile(r"^\[(\d+)-(\d+)\]$")

_RETIRED = {"RET", "ABD"}
_WALKOVER = {"W/O", "WO"}
_DEFAULT = {"DEF", "DEFAULT"}
_SUSPENDED = {"WEA"}  # play suspended (weather), seen in Davis Cup ties: unfinished, not rateable

STATUSES = ("completed", "retired", "walkover", "default", "suspended", "unknown")


@dataclass(frozen=True)
class ParsedScore:
    status: str              # one of STATUSES
    sets_w: int              # completed sets won by the match winner
    sets_l: int              # completed sets won by the match loser
    games_w: int             # games won by the winner (excludes match tiebreak)
    games_l: int
    tiebreaks: int           # set tiebreaks played (7-6 / 6-7 sets)
    match_tiebreak: bool     # decided by a [10-x] match tiebreak
    last_set_partial: bool   # final set unfinished (retirement mid-set)


def _set_complete(a: int, b: int, has_tb: bool) -> bool:
    hi, lo = max(a, b), min(a, b)
    return has_tb or (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo in (5, 6))


def _empty(status: str) -> ParsedScore:
    return ParsedScore(status, 0, 0, 0, 0, 0, False, False)


def parse_score(score: Optional[str]) -> ParsedScore:
    if score is None or not str(score).strip():
        return _empty("unknown")

    tokens = str(score).strip().upper().split()
    flag: Optional[str] = None
    sets: list[tuple[int, int, bool]] = []
    match_tb = False

    for tok in tokens:
        if tok in _WALKOVER:
            flag = "walkover"
        elif tok in _DEFAULT:
            flag = "default"
        elif tok in _RETIRED:
            flag = "retired"
        elif tok in _SUSPENDED:
            flag = "suspended"
        elif (m := _MATCH_TB_RE.match(tok)):
            match_tb = True
            # a match tiebreak is a set won by the player who has more points
            a, b = int(m.group(1)), int(m.group(2))
            sets.append((a, b, None))  # type: ignore[arg-type]
        elif (m := _SET_RE.match(tok)):
            sets.append((int(m.group(1)), int(m.group(2)), m.group(3) is not None))
        else:
            return _empty("unknown")

    if flag == "walkover" and not sets:
        return _empty("walkover")
    if not sets:
        return _empty("unknown")

    sets_w = sets_l = games_w = games_l = tiebreaks = 0
    last_partial = False
    for i, (a, b, has_tb) in enumerate(sets):
        is_mtb = has_tb is None
        if is_mtb:
            complete = True
        else:
            complete = _set_complete(a, b, has_tb)
            games_w += a
            games_l += b
            if has_tb or (max(a, b) == 7 and min(a, b) == 6):
                tiebreaks += 1
        if complete:
            if a > b:
                sets_w += 1
            elif b > a:
                sets_l += 1
        elif i == len(sets) - 1:
            last_partial = True

    status = flag if flag in ("retired", "default", "suspended") else "completed"
    if flag == "walkover":  # sets present but flagged W/O: treat as unknown rather than guess
        status = "unknown"
    return ParsedScore(status, sets_w, sets_l, games_w, games_l, tiebreaks, match_tb, last_partial)


def parsed_dict(score: Optional[str]) -> dict:
    return {"score": score, **asdict(parse_score(score))}
