# v0.5: four attempted improvements

Pre-stated rule: a step is kept only if it lowers rolling-origin log loss (2018-2022, each year predicted
from earlier years) with a tournament-clustered 95% interval excluding zero for **both** logistic regression
and LightGBM. Hyperparameters are the v0.4 ones; nothing here touches 2023+.

## Data added
Qualifying (2007-2026, 29,774 matches) and Challenger (2000-2026, 121,911 matches) results. They update
ratings only (weight `aux_k` = 0.75 of a normal match, tuned on 2008-2021) and are never predicted.

## Elo ladder (tuning 2008-2021, validation 2022)
| Step | Tune log loss | Validation 2022 | Δ vs previous kept |
|---|---|---|---|
| Elo v0.3 | 0.5950 | 0.6137 | |
| + rank-seeded start | 0.5952 | 0.6132 | no gain, dropped |
| + qualifying and Challenger results | 0.5869 | 0.6060 | -0.0077 [-0.0150, -0.0009] |

## Feature steps (rolling origin 2018-2022; n = 11,650)
| Step | LR Δ vs previous | GBM Δ vs previous | Decision |
|---|---|---|---|
| Elo learns from qualifying/Challenger | -0.0033 [-0.0053, -0.0015] | -0.0034 [-0.0053, -0.0014] | **keep** |
| + rating trend (90/365 d) | +0.0001 [-0.0001, +0.0004] | +0.0003 [-0.0005, +0.0010] | drop |
| + Glicko uncertainty | -0.0010 [-0.0016, -0.0005] | -0.0002 [-0.0010, +0.0006] | drop (LR only) |
| + serve/return point model | -0.0002 [-0.0004, +0.0001] | -0.0003 [-0.0010, +0.0003] | drop |

Net: LR 0.6046 -> 0.6013, GBM 0.6027 -> 0.5994 over 2018-22. Validation 2022: LR 0.5960 -> 0.5919,
GBM 0.5963 -> 0.5927. Full tables: `v05_ablation.md` / `.json`.

## Reading
* The only reliable gain came from better data for the ratings, not from new features.
* Trend, Glicko and point-model features carry information already in the Elo blend.
* Point-model tuning on training years preferred surface mix 0.5 and zero pace shift, i.e. court pace added nothing.

## Test-set note
The 2023+ test set was scored once for v0.4 (`test_results.json`). v0.5 was chosen using development years
only. Any v0.5 test score would be a second look and must use a separate marker file.
