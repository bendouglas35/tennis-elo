# tennis-elo

Predicting ATP tour-level match outcomes with Elo variants and gradient boosting, using DuckDB/SQL as
a leakage-safe feature layer. The aim is a carefully validated model with honest uncertainty, not a
claim of beating the betting market (closing lines are very sharp).

**Status: v0.1 in progress.** Data loading, score parsing, the symmetric modelling table and a
ranking baseline are built and tested on synthetic data. Real-data results are not in yet.

## Question

Does surface-aware, uncertainty-aware Elo, or a gradient-boosted model on lagged form and fatigue
features, predict ATP matches better than a ranking baseline, and by how much (with confidence
intervals)?

## Quick start

```bash
make setup                     # pip install -e ".[dev]"
make data                      # download atp_matches_2000..now into data/raw
#   or: make data FROM=/path/to/folder/with/atp_matches_YYYY.csv
make build                     # raw CSVs -> data/tennis.duckdb
make baseline                  # ranking baseline on train/validation years
make test
```

## Design decisions

- **Symmetric rows.** Each match is labelled player A / player B by a hash of `match_id`, so the
  winner is not always "player 1". The target is `a_wins`. See `sql/20_match_sides.sql`.
- **Leakage.** Columns from `minutes` to `l_bpFaced` describe the match itself. They only enter
  features through windows that end strictly before the match (v0.2), with automated tests.
- **Return stats are derived.** The source has no return columns; they are the opponent's serve
  stats inverted (`sql/25_player_matches.sql`).
- **Walkovers and defaults** are excluded from ratings and training. Retirements are kept and
  flagged; handling them in the Elo update is a v0.3 ablation.
- **Frozen split** (`src/tennis/__init__.py`): train <= 2021, validation 2022, test >= 2023. The
  test years are only evaluated at milestone points, never during development.

## Layout

```
sql/        numbered build steps (matches, symmetric sides, long player_matches)
src/tennis/ download, score parsing, build, metrics, baseline, CLI
tests/      score parsing, pipeline invariants, symmetry, determinism, split discipline
docs/       write-ups per milestone
```

## Roadmap (scoped down: the market is sharp, so this stays small)

| Step | Content |
|---|---|
| v0.1 | Data layer, symmetric table, ranking baseline |
| v0.2 | SQL feature view: rolling return/serve form, dominance ratio, rest and fatigue, leakage tests |
| v0.3 | Elo ladder: overall, surface-blended, dynamic K with layoff decay |
| v0.4 | Logistic regression and one GBM, calibration, bootstrap CIs, SHAP, error analysis |

## Results

Pending real data. This table will hold log loss / Brier per model with bootstrap CIs.

## Data and licence

Two interchangeable sources with the same core columns. Use **one per raw directory** (mixing is
refused, since it would duplicate matches):

- **TennisMyLife** (<https://stats.tennismylife.org/tennis-match-database>): `YYYY.csv`, ATP player
  IDs, an extra `indoor` column, updated daily. Stated licence: MIT; it says it is enriched from
  other sources, so check its terms before redistributing anything derived.
  `make data` with `--source tml`, or drop the CSVs in and use `FROM=`.
- **Jeff Sackmann's tennis_atp** (<https://github.com/JeffSackmann/tennis_atp>): `atp_matches_YYYY.csv`,
  licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial,
  attribution, share-alike). Credit: Tennis databases, files, and algorithms by Jeff Sackmann /
  Tennis Abstract.

Player IDs differ between the two sources, so their files cannot be joined on `player_id`.
The source changes daily, so **pin a snapshot** (date and file hashes) before evaluating on the
frozen test years. Raw data and derived databases are git-ignored and not redistributed.
