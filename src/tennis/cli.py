from __future__ import annotations

import argparse
from pathlib import Path

from . import baseline
from .build import build
from .download import copy_from_dir, download


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tennis")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch atp_matches_YYYY.csv into data/raw")
    d.add_argument("--raw-dir", default="data/raw")
    d.add_argument("--first-year", type=int, default=2000)
    d.add_argument("--source", choices=["sackmann", "tml"], default="sackmann")
    d.add_argument("--from-dir", help="copy CSVs from a local folder instead of downloading")

    b = sub.add_parser("build", help="build the DuckDB database from raw CSVs")
    b.add_argument("--raw-dir", default="data/raw")
    b.add_argument("--db", default="data/tennis.duckdb")
    b.add_argument("--min-year", type=int, default=2000)

    r = sub.add_parser("baseline", help="ranking baseline on train/validation years")
    r.add_argument("--db", default="data/tennis.duckdb")

    a = p.parse_args(argv)
    if a.cmd == "download":
        if a.from_dir:
            copy_from_dir(Path(a.from_dir), Path(a.raw_dir), a.first_year)
        else:
            download(Path(a.raw_dir), a.first_year, source=a.source)
    elif a.cmd == "build":
        for k, v in build(Path(a.db), Path(a.raw_dir), a.min_year).items():
            print(f"{k:>16}: {v:,}")
    elif a.cmd == "baseline":
        res = baseline.run(a.db)
        print(f"train n={res['n_train']:,}  validation n={res['n_val']:,}")
        print(f"higher-ranked-wins accuracy (val): {res['val_accuracy_higher_ranked']:.4f}")
        print(f"logistic on log rank ratio (val):  acc={res['val_accuracy_logistic']:.4f}  "
              f"log loss={res['val_log_loss']:.4f}  brier={res['val_brier']:.4f}")


if __name__ == "__main__":
    main()
