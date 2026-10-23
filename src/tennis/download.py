"""Fetch Jeff Sackmann's tour-level ATP match files (CC BY-NC-SA 4.0).

Credit: Tennis databases, files, and algorithms by Jeff Sackmann / Tennis Abstract,
https://github.com/JeffSackmann/tennis_atp. Non-commercial use, attribution required,
derived datasets must be shared alike. Raw files are git-ignored, not redistributed.
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path

SOURCES = {
    # name -> (url template, saved file name template)
    "sackmann": ("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
                 "atp_matches_{year}.csv"),
    # TennisMyLife (https://stats.tennismylife.org/tennis-match-database): same columns plus `indoor`.
    "tml": ("https://stats.tennismylife.org/data/{year}.csv", "{year}.csv"),
}
# Tour-level main-draw files: Sackmann's atp_matches_YYYY.csv or TennisMyLife's YYYY.csv.
# Qualifying / challenger / futures files have other names and are not matched.
MATCH_FILE_RE = re.compile(r"^(?:atp_matches_)?(\d{4})\.csv$")


def match_files(raw_dir: Path, min_year: int = 0) -> list[Path]:
    out, seen = [], {}
    for p in sorted(Path(raw_dir).glob("*.csv")):
        m = MATCH_FILE_RE.match(p.name)
        if not m or int(m.group(1)) < min_year:
            continue
        year = int(m.group(1))
        if year in seen:
            raise SystemExit(
                f"Year {year} appears twice ({seen[year].name}, {p.name}): mixing sources would "
                "duplicate matches. Keep one source per raw directory."
            )
        seen[year] = p
        out.append(p)
    return out


def download(raw_dir: Path, first_year: int = 2000, last_year: int | None = None,
             source: str = "sackmann") -> list[Path]:
    url_tpl, name_tpl = SOURCES[source]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_year = last_year or dt.date.today().year
    got: list[Path] = []
    for year in range(first_year, last_year + 1):
        name = name_tpl.format(year=year)
        url = url_tpl.format(year=year)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            print(f"  skip {name}: HTTP {e.code}")
            continue
        except urllib.error.URLError as e:
            print(f"  skip {name}: {e.reason}")
            continue
        (raw_dir / name).write_bytes(data)
        got.append(raw_dir / name)
        print(f"  ok   {name} ({len(data):,} bytes)")
    if not got:
        raise SystemExit(
            "No match files downloaded. If the host is unreachable from here, download the "
            "yearly CSVs manually and run: tennis download --from-dir <folder>"
        )
    return got


def copy_from_dir(src_dir: Path, raw_dir: Path, min_year: int = 0) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = match_files(Path(src_dir), min_year)
    if not files:
        raise SystemExit(f"No yearly match CSVs (atp_matches_YYYY.csv or YYYY.csv) found in {src_dir}")
    out = []
    for f in files:
        shutil.copy2(f, raw_dir / f.name)
        out.append(raw_dir / f.name)
    print(f"copied {len(out)} files into {raw_dir}")
    return out
