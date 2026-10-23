"""Build the DuckDB database from raw Sackmann CSVs: raw -> matches -> match_sides / player_matches."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .download import match_files
from .score import parsed_dict

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
SQL_STEPS = ["10_matches.sql", "20_match_sides.sql", "25_player_matches.sql"]


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def build(db_path: Path, raw_dir: Path, min_year: int = 2000) -> dict:
    files = match_files(Path(raw_dir), min_year)
    if not files:
        raise SystemExit(f"No atp_matches_YYYY.csv files >= {min_year} in {raw_dir}. Run `make data` first.")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        file_list = ", ".join(_quote(f) for f in files)
        con.execute(
            f"create or replace table raw_matches as "
            f"select * from read_csv([{file_list}], header=true, all_varchar=true, union_by_name=true)"
        )

        # TennisMyLife files carry an `indoor` column; Sackmann's do not. Keep one schema.
        cols = {r[0].lower() for r in con.execute("describe raw_matches").fetchall()}
        if "indoor" not in cols:
            con.execute("alter table raw_matches add column indoor varchar")

        scores = [r[0] for r in con.execute("select distinct score from raw_matches").fetchall()]
        score_parsed = pd.DataFrame([parsed_dict(s) for s in scores])
        con.register("score_parsed", score_parsed)

        for step in SQL_STEPS:
            con.execute((SQL_DIR / step).read_text())

        dupes = con.execute(
            "select count(*) from (select match_id from matches group by 1 having count(*) > 1)"
        ).fetchone()[0]
        if dupes:
            raise SystemExit(f"{dupes} duplicate match_id values in matches; inspect tourney_id/match_num.")

        counts = {
            "files": len(files),
            "matches": con.execute("select count(*) from matches").fetchone()[0],
            "rateable": con.execute("select count(*) from matches where is_rateable").fetchone()[0],
            "match_sides": con.execute("select count(*) from match_sides").fetchone()[0],
            "player_matches": con.execute("select count(*) from player_matches").fetchone()[0],
            "unknown_scores": con.execute(
                "select count(*) from matches where score_status = 'unknown'").fetchone()[0],
        }
        return counts
    finally:
        con.close()
