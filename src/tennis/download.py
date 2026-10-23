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

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
MATCH_FILE_RE = re.compile(r"^atp_matches_(\d{4})\.csv$")


def match_files(raw_dir: Path, min_year: int = 0) -> list[Path]:
    """Tour-level main-draw files only (qual/challenger/futures files don't match the pattern)."""
    out = []
    for p in sorted(Path(raw_dir).glob("atp_matches_*.csv")):
        m = MATCH_FILE_RE.match(p.name)
        if m and int(m.group(1)) >= min_year:
            out.append(p)
    return out


def download(raw_dir: Path, first_year: int = 2000, last_year: int | None = None,
             base_url: str = BASE_URL) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_year = last_year or dt.date.today().year
    got: list[Path] = []
    for year in range(first_year, last_year + 1):
        name = f"atp_matches_{year}.csv"
        url = f"{base_url}/{name}"
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
            "No match files downloaded. If GitHub is unreachable from here, download "
            "atp_matches_YYYY.csv manually and run: tennis download --from-dir <folder>"
        )
    return got


def copy_from_dir(src_dir: Path, raw_dir: Path, min_year: int = 0) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = match_files(Path(src_dir), min_year)
    if not files:
        raise SystemExit(f"No atp_matches_YYYY.csv files found in {src_dir}")
    out = []
    for f in files:
        shutil.copy2(f, raw_dir / f.name)
        out.append(raw_dir / f.name)
    print(f"copied {len(out)} files into {raw_dir}")
    return out
