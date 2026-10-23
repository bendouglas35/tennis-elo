"""Synthetic Sackmann-format CSVs so the pipeline is testable without the real data."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

COLUMNS = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,"
    "winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,"
    "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,loser_ioc,loser_age,"
    "score,best_of,round,minutes,"
    "w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,"
    "l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,"
    "winner_rank,winner_rank_points,loser_rank,loser_rank_points"
).split(",")


def make_row(year: int, n: int, winner: int, loser: int, score: str = "6-3 6-4",
             wrank: int | None = 10, lrank: int | None = 50, **over) -> dict:
    row = {c: "" for c in COLUMNS}
    row.update(
        tourney_id=f"{year}-T{n // 20}", tourney_name=f"Open {n // 20}", surface="Hard",
        draw_size="32", tourney_level="A", tourney_date=f"{year}0110", match_num=str(n),
        winner_id=str(winner), winner_name=f"P{winner}", winner_hand="R", winner_ht="185",
        winner_ioc="GBR", winner_age="25.0",
        loser_id=str(loser), loser_name=f"P{loser}", loser_hand="R", loser_ht="183",
        loser_ioc="ESP", loser_age="26.0",
        score=score, best_of="3", round="R32", minutes="90",
        w_ace="5", w_df="2", w_svpt="60", w_1stIn="40", w_1stWon="30", w_2ndWon="10",
        w_SvGms="9", w_bpSaved="2", w_bpFaced="3",
        l_ace="3", l_df="3", l_svpt="70", l_1stIn="42", l_1stWon="28", l_2ndWon="12",
        l_SvGms="9", l_bpSaved="3", l_bpFaced="7",
        winner_rank="" if wrank is None else str(wrank), winner_rank_points="2000",
        loser_rank="" if lrank is None else str(lrank), loser_rank_points="800",
    )
    row.update({k: str(v) for k, v in over.items()})
    return row


def write_csv(path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def stat_overrides(rng: random.Random) -> dict:
    """Coherent random serve stats for both players of a match."""
    out = {}
    for side, (lo, hi) in (("w", (0.62, 0.80)), ("l", (0.55, 0.72))):
        svpt = rng.randint(50, 110)
        first_in = int(svpt * rng.uniform(0.5, 0.7))
        first_won = int(first_in * rng.uniform(lo, hi))
        second_won = int((svpt - first_in) * rng.uniform(0.4, 0.6))
        out.update({f"{side}_svpt": svpt, f"{side}_1stIn": first_in, f"{side}_1stWon": first_won,
                    f"{side}_2ndWon": second_won, f"{side}_ace": rng.randint(0, 15),
                    f"{side}_df": rng.randint(0, 6), f"{side}_bpFaced": rng.randint(0, 9),
                    f"{side}_bpSaved": rng.randint(0, 4), f"{side}_SvGms": rng.randint(8, 16)})
    out["minutes"] = rng.randint(50, 220)
    return out


@pytest.fixture
def rich_raw_dir(tmp_path: Path) -> Path:
    """3 years x 26 weekly 16-player knockout events (R16..F) with varied stats; event ids repeat
    across years (tourney_id suffix), like the real source."""
    import datetime as dt

    rng = random.Random(1)
    d = tmp_path / "rich"
    d.mkdir()
    for year in (2019, 2020, 2021):
        rows = []
        for week in range(26):
            date = dt.date(year, 1, 4) + dt.timedelta(days=7 * week)
            alive = rng.sample(range(1, 41), 16)
            n = 0
            for rnd, count in (("R16", 8), ("QF", 4), ("SF", 2), ("F", 1)):
                nxt = []
                for i in range(count):
                    a, b = alive[2 * i], alive[2 * i + 1]
                    w, l = (min(a, b), max(a, b)) if rng.random() < 0.65 else (max(a, b), min(a, b))
                    n += 1
                    rows.append(make_row(
                        year, n, w, l, wrank=w, lrank=l,
                        tourney_id=f"{year}-{500 + week}", tourney_name=f"Event {500 + week}",
                        tourney_date=date.strftime("%Y%m%d"), round=rnd, match_num=n,
                        **stat_overrides(rng)))
                    nxt.append(w)
                alive = nxt
        write_csv(d / f"{year}.csv", rows)
    return d


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Years 2020-2022. Higher-ranked player always wins (rank = player id)."""
    rng = random.Random(0)
    d = tmp_path / "raw"
    d.mkdir()
    for year in (2020, 2021, 2022):
        rows = []
        for n in range(1, 201):
            a, b = rng.sample(range(1, 101), 2)
            w, l = (a, b) if a < b else (b, a)
            rows.append(make_row(year, n, w, l, wrank=w, lrank=l))
        # special cases, numbered above the generated ones
        rows.append(make_row(year, 901, 1, 2, score="6-3 2-0 RET", wrank=1, lrank=2))
        rows.append(make_row(year, 902, 3, 4, score="W/O", wrank=3, lrank=4,
                             w_ace="", w_svpt="", l_svpt="", w_1stWon="", w_2ndWon="",
                             l_1stWon="", l_2ndWon=""))
        rows.append(make_row(year, 903, 5, 6, score="6-4 6-4", wrank=None, lrank=None))
        write_csv(d / f"atp_matches_{year}.csv", rows)
    # files that must be ignored: challenger/qualifying and futures-style names
    write_csv(d / "atp_matches_qual_chall_2021.csv", [make_row(2021, 1, 7, 8)])
    write_csv(d / "atp_matches_futures_2021.csv", [make_row(2021, 1, 7, 8)])
    return d
