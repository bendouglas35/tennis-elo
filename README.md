# tennis-elo

Predicting ATP tour-level match outcomes with Elo variants and gradient boosting, using DuckDB/SQL as
a leakage-safe feature layer. The aim is a carefully validated model with honest uncertainty, not a
claim of beating the betting market (closing lines are very sharp).

**Status: v0.2 feature layer complete.** Data layer, symmetric modelling table, ranking baseline,
and the leakage-safe feature layer (form, fatigue, experience, entry, court pace) are built and
tested on real data. The Elo ladder and ML models follow.

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
make leakage                   # truncation + perturbation leakage checks on the built database
make pace-tune                 # re-tune court-pace smoothing on training years
make test
```

## Design decisions

- **Symmetric rows.** Each match is labelled player A / player B by a hash of `match_id`, so the
  winner is not always "player 1". The target is `a_wins`. See `sql/20_match_sides.sql`.
- **Leakage.** Columns from `minutes` to `l_bpFaced` describe the match itself. They only enter
  features through windows that end strictly before the match, ordered by `ord_num` (tournament
  week, then round; see `sql/10_matches.sql`). `make leakage` deletes the future and overwrites
  in-match stats (one week at a time) and checks that no feature moves; the tests also inject a deliberate leak to prove
  the checks fail when they should.
- **Court pace** is keyed by event, surface and indoor/outdoor: per-match ace and serve-points-won
  residuals against the players' own lagged records, pooled over the event's previous editions, plus a
  running in-tournament value. Tuned on training years only: it explains most of the edition-level
  variation that surface alone misses (error 0.100 -> 0.060). Details in `docs/feature_plan.md`.
- **Return stats are derived.** The source has no return columns; they are the opponent's serve
  stats inverted (`sql/25_player_matches.sql`).
- **Walkovers and defaults** are excluded from ratings and training. Retirements are kept and
  flagged; handling them in the Elo update is a v0.3 ablation.
- **Frozen split** (`src/tennis/__init__.py`): train <= 2021, validation 2022, test >= 2023. The
  test years are only evaluated at milestone points, never during development.

## Layout

```
sql/        numbered build steps (matches, symmetric sides, long player_matches)
src/tennis/ download, score parsing, build, features and court pace (SQL generators), pace tuning,
            leakage checks, metrics, baseline, CLI
tests/      score parsing, pipeline invariants, symmetry, determinism, split discipline
docs/       write-ups per milestone
```

## Roadmap (scoped down: the market is sharp, so this stays small)

| Step | Content |
|---|---|
| v0.1 | Data layer, symmetric table, ranking baseline |
| v0.2 | SQL feature layer: form, dominance ratio, rest and fatigue, court pace, leakage tests (built) |
| v0.3 | Elo ladder: overall, surface-blended, dynamic K with layoff decay |
| v0.4 | Logistic regression and one GBM, calibration, bootstrap CIs, SHAP, error analysis |

## Results

Population: ordinary tour matches (Davis Cup, Laver Cup, ATP Cup, United Cup and Next Gen Finals
excluded). Train 2000-2021, validated on 2022 (n = 2,625). The frozen test years (2023+) have not
been evaluated. Data snapshot: `docs/data_snapshot.md`. A coin flip scores log loss 0.6931.

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Higher-ranked player wins | 0.6408 | n/a (hard prediction) | n/a |
| Logistic on log rank ratio (v0.1 baseline) | 0.6408 | 0.6314 | 0.2205 |

Elo variants and the ML models will be added here with bootstrap confidence intervals. Including
team events makes the same baseline look slightly better (log loss 0.6260), because those matches
are easier to predict.

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
