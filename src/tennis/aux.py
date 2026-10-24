"""Auxiliary results (ATP qualifying, Challenger tour) used only to rate players with Elo/Glicko.

They never become prediction targets and never enter the SQL form/fatigue windows, so the v0.1-v0.4
feature layer and its leakage tests are untouched. They share players' ATP-style ids with the main
files (TennisMyLife source), which is what lets a qualifier's or Challenger player's rating carry
over when he reaches the main draw.

Ordering uses the same key as `matches` (see sql/10_matches.sql). Qualifying rounds happen before
the main draw of the same week, so Q1 < Q2 < Q3 get round ranks -2, -1, 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from .score import parsed_dict

RAW_DIRS = {"quali": Path("data/raw_quali"), "chall": Path("data/raw_chall")}
_YEAR = re.compile(r"(\d{4})(?:_\w+)?\.csv$")


def _files(d: Path):
    return sorted(f for f in Path(d).glob("*.csv") if _YEAR.search(f.name))


def build_aux(db_path, raw_dirs: dict | None = None, verbose: bool = True) -> dict:
    raw_dirs = raw_dirs or RAW_DIRS
    con = duckdb.connect(str(db_path))
    try:
        frames = []
        for source, d in raw_dirs.items():
            files = _files(d) if Path(d).exists() else []
            if not files:
                continue
            lst = ", ".join("'" + str(f).replace("'", "''") + "'" for f in files)
            frames.append(f"select '{source}' as source, * from read_csv([{lst}], header=true, "
                          f"all_varchar=true, union_by_name=true)")
        if not frames:
            raise SystemExit("No auxiliary CSVs found (data/raw_quali, data/raw_chall).")
        con.execute("create or replace table aux_raw as " + " union all by name ".join(frames))
        scores = [r[0] for r in con.execute("select distinct score from aux_raw").fetchall()]
        con.register("aux_scores", pd.DataFrame([parsed_dict(s) for s in scores]))
        con.execute("""
        create or replace table aux_matches as
        with t as (
            select r.*, try_strptime(r.tourney_date, '%Y%m%d')::date as d, s.status, s.games_w, s.games_l
            from aux_raw r left join aux_scores s on r.score is not distinct from s.score
        )
        select source || '|' || tourney_id || '|' || coalesce(match_num, coalesce(round, '?') || '-' || winner_id
               || '-' || loser_id) as match_id,
               source, tourney_id, split_part(tourney_id, '-', 2) as event_id, tourney_name, tourney_level,
               surface, d as tourney_date, round,
               try_cast(best_of as integer) as best_of,
               winner_id, loser_id,
               try_cast(winner_rank as integer) as winner_rank, try_cast(loser_rank as integer) as loser_rank,
               try_cast(winner_age as double) as winner_age, try_cast(loser_age as double) as loser_age,
               games_w, games_l, (status = 'retired') as retired,
               (((d - date '1970-01-01') * 16 + case round
                   when 'Q1' then -2 when 'Q2' then -1 when 'Q3' then 0
                   when 'R128' then 1 when 'R64' then 2 when 'R32' then 3 when 'R16' then 4
                   when 'QF' then 5 when 'SF' then 6 when 'F' then 7 else 4 end) * 1024)::bigint as ord_num
        from t where status in ('completed', 'retired') and d is not null
        """)
        dupes = con.execute("select count(*) from (select match_id from aux_matches group by 1 having count(*) > 1)").fetchone()[0]
        if dupes:
            raise SystemExit(f"{dupes} duplicate auxiliary match ids")
        out = dict(con.execute("select source, count(*) from aux_matches group by 1").fetchall())
        if verbose:
            print("aux_matches rows by source:", out)
        return out
    finally:
        con.close()
