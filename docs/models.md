# v0.4: models, calibration, error analysis, frozen test

Code: `src/tennis/dataset.py` (modelling frame), `models.py`, `experiments.py` (tuning),
`report.py` (tables, calibration, SHAP, slices, ablations). Commands: `make tune`, `make validate`,
`make final-test`. Raw numbers: `docs/v04_validation.json`, `docs/test_results.json`; figures in
`docs/figures/`.

## Protocol (fixed before looking at the test years)

- **Population:** standard tour matches from 2003 (2000-2002 is rating and form warm-up). Team events
  and the Next Gen Finals are excluded as targets.
- **Features:** the SQL layer (form, fatigue, entry, rank, court pace, altitude) plus the tuned Elo
  ratings from v0.3, all as A-minus-B differences. Pace and indoor/outdoor are symmetric context,
  so they enter through explicit interactions with the differences (for example pace x ace-rate
  difference); a main effect would cancel out of an A/B target.
- **Symmetry:** both models are exactly antisymmetric, p(x) + p(mirror x) = 1, tested. Logistic
  regression uses only antisymmetric columns with no intercept. LightGBM is trained on the data plus
  its mirror image and predicts the average of f(x) and 1 - f(mirror x).
- **Tuning** (logistic C; GBM leaves, leaf size, rounds) used expanding-window evaluation on 2018-2021
  (each year predicted by a model trained on earlier years). The first GBM pass picked an edge of the
  grid, so it was widened once (1,600 rounds, 32 leaves); the final choice is interior
  (16 leaves, 1,100 rounds, learning rate 0.03). A one-parameter recalibration (slope on the logit,
  no intercept) was fitted on the same out-of-fold predictions (slope 0.90).
- **Validation (2022)** was only scored. **The frozen test (2023 onwards, n = 10,412) was scored once**,
  after the code and `docs/model_config.json` were committed (commit 8a5d440), with models refitted on
  2003-2022 and nothing re-tuned. `make final-test` refuses to run a second time.
- Intervals are 95% percentile intervals from a bootstrap that resamples whole tournaments (2,000
  draws); differences are paired on the same draws.

## Results

Validation 2022 and the frozen test, log loss (lower is better; a coin flip is 0.6931):

| Model | Validation 2022 | Frozen test 2023+ |
|---|---|---|
| Rank baseline | 0.6318 | 0.6314 [0.6214, 0.6418] |
| Elo (tuned, v0.3) | 0.6131 | 0.6258 [0.6152, 0.6366] |
| Logistic regression, all features | 0.5960 | 0.6095 [0.5979, 0.6210] |
| LightGBM, all features, recalibrated | 0.5967 | 0.6069 [0.5961, 0.6175] |
| Average of the two | 0.5950 | 0.6066 [0.5956, 0.6175] |

On the test set the best model beats the ranking baseline by 0.0247 log loss (interval 0.0200 to 0.0294)
and the tuned Elo by 0.0192 (0.0158 to 0.0226). Accuracy is 65.9% against 63.8% for "higher-ranked
player wins" and 64.2% for Elo. The gains held in every test year (2026 is a partial season):
GBM 0.608 / 0.597 / 0.616 / 0.608 for 2023-2026 against Elo 0.627 / 0.618 / 0.629 / 0.630.

**What this does and does not show.** The models are clearly better than ranks and than Elo, and the
improvement is stable out of sample. But the gap between a carefully tuned Elo and a 100-feature gradient
booster is about 0.02 log loss, which is 3% of the loss, and the logistic regression is as good as
the GBM within noise. That is what a mostly-noise outcome looks like. Nothing here is compared with
betting-market prices, so it says nothing about being able to beat the market.

## Calibration

Reliability plots: `docs/figures/calibration_validation.png` and `calibration_test.png`. Raw Elo is
the worst calibrated on the test set (expected calibration error 0.029), the raw GBM is next (0.026),
and the slope-recalibrated GBM is best (0.013); the figure shows the direction of the misses. The slope
was fitted on 2018-2021 out-of-fold predictions and carried to the test unchanged.

## Court pace: does it earn its place?

Rolling-origin ablation over 2018-2022 (n = 11,650), each year predicted from earlier years only:

| Step (log loss change, 95% CI) | Logistic regression | LightGBM |
|---|---|---|
| + indoor, altitude | -0.0001 [-0.0003, +0.0001] | -0.0001 [-0.0008, +0.0006] |
| + static pace | **-0.0007** [-0.0012, -0.0002] | -0.0006 [-0.0014, +0.0002] |
| + running in-tournament pace | -0.0000 [-0.0002, +0.0001] | -0.0000 [-0.0008, +0.0007] |

- **Static pace is a real but tiny effect:** about 0.0007 log loss, significant for the logistic
  regression only. Its interaction with the ace-rate difference has the expected sign (a serving edge
  matters more on a fast court). On the frozen test, adding pace to the logistic regression changes
  log loss by -0.0002 (0.6097 to 0.6095), which is within noise.
- **Running pace, indoor/outdoor and altitude add nothing detectable.** I kept them in the feature
  set because the plan said to test them, not because they helped, and a reader should treat them
  as ablated away.
- Pace is strongly predictable as a property of the event (v0.2: error cut by 40% against the surface
  mean), but a match outcome is mostly decided by which player is better, so the court barely moves
  the probability. This is the honest ceiling for a single-variable "court" feature.
- Caveat: the ablation years overlap the tuning years (2018-2021), so absolute levels are slightly
  optimistic; the paired differences are what matter.

## Which features matter

Dropping a group from the full model (rolling origin, log loss increase, LightGBM): form +0.0041,
biographical (age, height, handedness) +0.0042, Elo +0.0026, rank +0.0017, fatigue +0.0013, entry
+0.0006 (not significant). SHAP (`docs/figures/shap_bar_validation.png`, `shap_swarm_validation.png`)
puts the Elo blend first and then age difference, surface-specific form sample size, overall Elo,
rank points and recent win rate.

Two cautions. **Age** is the second-largest SHAP feature (younger players do better, other things
equal). That is plausible, but age is entangled with experience and rating lag (young players improve
faster than Elo can track), so it may partly be standing in for a better rating update rather than
for an age effect. **Several feature groups overlap** (Elo, rank, form), so the one-group-out drops
understate what each contributes alone.

## Error analysis

Full slices are in the two result files below. Patterns on the test set (validation is similar but noisier):

- The model's gain over Elo is largest where Elo is least informed: players with fewer than 20 rated
  matches (-0.030 on test), best-of-five matches (-0.028), Grand Slam matches (-0.028), early rounds
  (R128 -0.032) and clay (-0.025). It is smallest in quarter-finals and semi-finals (-0.003 to -0.004),
  where the field is the strongest and closest.
- Close matches (rank gap below 0.7 in log terms) sit at 57-58% accuracy on the test set; there the model is hardly
  better than a coin flip with a small tilt. Matches with a large rank gap are 75% accurate.
- 250-level events are the hardest to predict (log loss 0.640, accuracy 62.7%).
- Slices with fewer than about 70 matches (the "A" level, finals, round robin) move around and
  should not be read.
- The largest confident misses on validation are lower-ranked players beating clear favourites
  (a table is in the validation section below). I did not find a systematic failure, but I only looked at ten.

## Limitations

- Elo was tuned on 2008-2021 and the other hyperparameters on 2018-2021, so those years are not clean for
  absolute performance; validation 2022 and the test years are.
- The test refit uses 2022 as training data while the hyperparameters were chosen without it, which is
  standard but means validation numbers and test numbers are not the same models.
- Tournament-clustered intervals do not cover dependence across matches of the same player over time.
- Data: TennisMyLife files as downloaded when the snapshot was taken, 2026 partial, 2020 about half
  size. See `docs/data_snapshot.md`; files change daily, so re-pin before quoting numbers.

---

## Validation 2022 (models fitted on 2003-2021, n = 2,630)

| Model | Log loss (95% CI) | Brier | Accuracy | Δ log loss vs Elo | Δ vs rank baseline |
|---|---|---|---|---|---|
| Rank baseline | 0.6318 [0.6133, 0.6526] | 0.2206 [0.2125, 0.2297] | 0.640 [0.616, 0.663] | 0.0187 [0.0084, 0.0286] |  |
| Elo (tuned) | 0.6131 [0.5925, 0.6363] | 0.2129 [0.2039, 0.2232] | 0.654 [0.626, 0.679] |  | -0.0187 [-0.0286, -0.0084] |
| LR: rank + Elo | 0.6121 [0.5926, 0.6339] | 0.2121 [0.2037, 0.2219] | 0.663 [0.637, 0.685] | -0.0010 [-0.0041, 0.0018] | -0.0197 [-0.0273, -0.0119] |
| LR: all features, no pace | 0.5973 [0.5785, 0.6203] | 0.2060 [0.1979, 0.2160] | 0.673 [0.647, 0.694] | -0.0158 [-0.0243, -0.0078] | -0.0345 [-0.0439, -0.0247] |
| LR: all features | 0.5960 [0.5770, 0.6189] | 0.2054 [0.1972, 0.2154] | 0.675 [0.649, 0.696] | -0.0171 [-0.0260, -0.0090] | -0.0358 [-0.0453, -0.0259] |
| GBM: all features | 0.5963 [0.5766, 0.6207] | 0.2056 [0.1969, 0.2160] | 0.675 [0.647, 0.698] | -0.0168 [-0.0270, -0.0072] | -0.0355 [-0.0459, -0.0242] |
| GBM: all features, recalibrated | 0.5967 [0.5781, 0.6198] | 0.2057 [0.1975, 0.2156] | 0.675 [0.647, 0.698] | -0.0164 [-0.0259, -0.0074] | -0.0351 [-0.0447, -0.0247] |
| Average of LR and GBM | 0.5950 [0.5762, 0.6179] | 0.2050 [0.1968, 0.2149] | 0.672 [0.645, 0.693] | -0.0181 [-0.0274, -0.0098] | -0.0368 [-0.0462, -0.0267] |

Expected calibration error (10 bins): Rank baseline 0.013, Elo (tuned) 0.017, LR: rank + Elo 0.015, LR: all features, no pace 0.008, LR: all features 0.012, GBM: all features 0.009, GBM: all features, recalibrated 0.011, Average of LR and GBM 0.007

GBM recalibration slope (fitted on 2018-2021 out-of-fold predictions): 0.899

### Error slices (GBM recalibrated)

| Slice | n | Accuracy | Log loss | Elo log loss | Rank log loss | Model − Elo |
|---|---|---|---|---|---|---|
| surface = Clay | 776 | 0.668 | 0.5941 | 0.6216 | 0.6198 | -0.0275 |
| surface = Grass | 319 | 0.680 | 0.6017 | 0.6058 | 0.6400 | -0.0041 |
| surface = Hard | 1535 | 0.678 | 0.5970 | 0.6103 | 0.6362 | -0.0133 |
| tourney_level = 250 | 1150 | 0.630 | 0.6355 | 0.6530 | 0.6577 | -0.0175 |
| tourney_level = 500 | 429 | 0.713 | 0.5883 | 0.6005 | 0.6279 | -0.0123 |
| tourney_level = G | 504 | 0.756 | 0.5165 | 0.5214 | 0.5516 | -0.0049 |
| tourney_level = M | 517 | 0.673 | 0.5952 | 0.6234 | 0.6503 | -0.0283 |
| round = F | 69 | 0.623 | 0.6888 | 0.7100 | 0.7259 | -0.0213 |
| round = QF | 262 | 0.637 | 0.6238 | 0.6275 | 0.6445 | -0.0036 |
| round = R128 | 320 | 0.728 | 0.5479 | 0.5748 | 0.5906 | -0.0269 |
| round = R16 | 529 | 0.671 | 0.5991 | 0.6104 | 0.6264 | -0.0114 |
| round = R32 | 909 | 0.661 | 0.6161 | 0.6345 | 0.6526 | -0.0184 |
| round = R64 | 380 | 0.729 | 0.5459 | 0.5622 | 0.5904 | -0.0163 |
| round = SF | 137 | 0.635 | 0.6146 | 0.6300 | 0.6358 | -0.0154 |
| best_of = 3.0 | 2111 | 0.656 | 0.6166 | 0.6354 | 0.6507 | -0.0188 |
| best_of = 5.0 | 519 | 0.753 | 0.5159 | 0.5223 | 0.5548 | -0.0064 |
| rank gap = <0.3 | 470 | 0.594 | 0.6591 | 0.6699 | 0.6906 | -0.0108 |
| rank gap = 0.3-0.7 | 620 | 0.627 | 0.6570 | 0.6723 | 0.6880 | -0.0153 |
| rank gap = 0.7-1.2 | 579 | 0.634 | 0.6246 | 0.6368 | 0.6664 | -0.0122 |
| rank gap = >1.2 | 956 | 0.770 | 0.5125 | 0.5341 | 0.5452 | -0.0216 |
| min Elo matches = <20 | 761 | 0.677 | 0.5905 | 0.6128 | 0.6215 | -0.0223 |
| min Elo matches = 20-100 | 1392 | 0.672 | 0.6011 | 0.6203 | 0.6436 | -0.0192 |
| min Elo matches = >100 | 477 | 0.681 | 0.5940 | 0.5924 | 0.6140 | +0.0015 |
| pace = slow | 886 | 0.691 | 0.5761 | 0.6106 | 0.6160 | -0.0346 |
| pace = medium | 1363 | 0.662 | 0.6109 | 0.6222 | 0.6385 | -0.0112 |
| pace = fast | 381 | 0.688 | 0.5940 | 0.5864 | 0.6446 | +0.0076 |

### Most confident misses (GBM recalibrated)

|    | tourney_name       | round   | winner_name        | loser_name         |   p_winner |
|---:|:-------------------|:--------|:-------------------|:-------------------|-----------:|
|  0 | US Open            | R128    | Brandon Holt       | Taylor Fritz       |      0.041 |
|  1 | Astana             | R32     | David Goffin       | Carlos Alcaraz     |      0.075 |
|  2 | Cincinnati Masters | R32     | Ben Shelton        | Casper Ruud        |      0.079 |
|  3 | Montpellier        | F       | Alexander Bublik   | Alexander Zverev   |      0.091 |
|  4 | US Open            | R128    | Daniel Elahi Galan | Stefanos Tsitsipas |      0.095 |
|  5 | Atlanta            | R32     | Andres Martin      | Thanasi Kokkinakis |      0.096 |
|  6 | Queen's Club       | R32     | Ryan Peniston      | Casper Ruud        |      0.101 |
|  7 | Marseille          | QF      | Roman Safiullin    | Stefanos Tsitsipas |      0.104 |
|  8 | 's-Hertogenbosch   | F       | Tim van Rijthoven  | Daniil Medvedev    |      0.114 |
|  9 | Australian Open    | R16     | Denis Shapovalov   | Alexander Zverev   |      0.118 |

### Logistic regression: largest standardised coefficients

|                          |   coefficient |
|:-------------------------|--------------:|
| d_log_rank_points        |         0.454 |
| d_f_win_rate             |        -0.341 |
| d_elo_overall            |         0.322 |
| d_fs_eff_stats           |         0.266 |
| d_f_game_win_pct         |         0.253 |
| d_elo_blend              |         0.245 |
| d_age                    |        -0.244 |
| d_f_dominance_ratio      |         0.221 |
| d_qualifier              |         0.168 |
| d_elo_surface            |         0.149 |
| d_n_28d                  |         0.130 |
| d_f_ret_won_pct          |        -0.126 |
| d_tourney_matches_before |         0.126 |
| d_fs_dominance_ratio     |         0.120 |
| d_log_rank               |        -0.114 |

Pace interactions (standardised):

|                |   coefficient |
|:---------------|--------------:|
| d_x_pace_s_ace |         0.062 |
| d_x_pace_s_ret |         0.029 |
| d_x_alt_ace    |         0.026 |
| d_x_pace_r_ace |         0.022 |
| d_x_pace_r_ret |        -0.015 |
| d_x_pace_s_sv  |         0.008 |
| d_x_indoor_sv  |        -0.008 |
| d_x_pace_r_sv  |        -0.003 |
| d_x_indoor_ace |        -0.001 |

### GBM mean |SHAP| (top 15)

|                          |   mean |SHAP| |
|:-------------------------|--------------:|
| d_elo_blend              |        0.3085 |
| d_age                    |        0.2065 |
| d_fs_eff_stats           |        0.1508 |
| d_elo_overall            |        0.1185 |
| d_log_rank_points        |        0.1042 |
| d_f_win_rate             |        0.0923 |
| d_elo_surface            |        0.0877 |
| d_f_game_win_pct         |        0.0755 |
| d_n_28d                  |        0.0713 |
| d_f_dominance_ratio      |        0.0586 |
| d_fs_game_win_pct        |        0.0557 |
| d_qualifier              |        0.0548 |
| d_fs_dominance_ratio     |        0.0380 |
| d_tourney_minutes_before |        0.0372 |
| d_log_rank               |        0.0347 |

### Ablation: LR

**LR**, rolling origin 2018-2022 (n = 11,650); each year is predicted by a model trained on all earlier years.

| Feature set | Log loss (95% CI) | Δ vs full model |
|---|---|---|
| surface only (no indoor, altitude or pace) | 0.6055 [0.5953, 0.6166] | +0.0008 [+0.0003, +0.0014] |
| + indoor, altitude | 0.6053 [0.5952, 0.6165] | +0.0007 [+0.0002, +0.0012] |
| + static pace | 0.6047 [0.5944, 0.6158] | +0.0000 [-0.0001, +0.0002] |
| + running pace (full) | 0.6046 [0.5944, 0.6157] |  |
| full without elo | 0.6081 [0.5982, 0.6189] | +0.0035 [+0.0018, +0.0053] |
| full without form | 0.6078 [0.5978, 0.6187] | +0.0032 [+0.0012, +0.0052] |
| full without fatigue | 0.6071 [0.5967, 0.6184] | +0.0025 [+0.0013, +0.0037] |
| full without rank | 0.6065 [0.5965, 0.6173] | +0.0019 [+0.0004, +0.0033] |
| full without entry | 0.6056 [0.5956, 0.6169] | +0.0010 [-0.0004, +0.0024] |
| full without bio | 0.6083 [0.5984, 0.6193] | +0.0037 [+0.0022, +0.0051] |

Pace steps, each against the previous one:

| Step | Δ log loss (95% CI) |
|---|---|
| + indoor, altitude | -0.0001 [-0.0003, +0.0001] |
| + static pace | -0.0007 [-0.0012, -0.0002] |
| + running pace (full) | -0.0000 [-0.0002, +0.0001] |

### Ablation: GBM

**GBM**, rolling origin 2018-2022 (n = 11,650); each year is predicted by a model trained on all earlier years.

| Feature set | Log loss (95% CI) | Δ vs full model |
|---|---|---|
| surface only (no indoor, altitude or pace) | 0.6035 [0.5932, 0.6147] | +0.0007 [-0.0002, +0.0016] |
| + indoor, altitude | 0.6034 [0.5931, 0.6147] | +0.0006 [-0.0002, +0.0015] |
| + static pace | 0.6028 [0.5927, 0.6139] | +0.0000 [-0.0007, +0.0008] |
| + running pace (full) | 0.6027 [0.5927, 0.6140] |  |
| full without elo | 0.6054 [0.5955, 0.6168] | +0.0026 [+0.0012, +0.0042] |
| full without form | 0.6068 [0.5964, 0.6179] | +0.0041 [+0.0020, +0.0059] |
| full without fatigue | 0.6041 [0.5939, 0.6155] | +0.0013 [+0.0002, +0.0025] |
| full without rank | 0.6044 [0.5944, 0.6155] | +0.0017 [+0.0004, +0.0029] |
| full without entry | 0.6033 [0.5932, 0.6147] | +0.0006 [-0.0007, +0.0019] |
| full without bio | 0.6069 [0.5968, 0.6184] | +0.0042 [+0.0027, +0.0056] |

Pace steps, each against the previous one:

| Step | Δ log loss (95% CI) |
|---|---|
| + indoor, altitude | -0.0001 [-0.0008, +0.0006] |
| + static pace | -0.0006 [-0.0014, +0.0002] |
| + running pace (full) | -0.0000 [-0.0008, +0.0007] |

---

## Frozen test 2023+ (models fitted on 2003-2022, n = 10,412)

| Model | Log loss (95% CI) | Brier | Accuracy | Δ log loss vs Elo | Δ vs rank baseline |
|---|---|---|---|---|---|
| Rank baseline | 0.6314 [0.6214, 0.6418] | 0.2209 [0.2166, 0.2255] | 0.638 [0.627, 0.648] | 0.0055 [0.0003, 0.0109] |  |
| Elo (tuned) | 0.6258 [0.6152, 0.6366] | 0.2185 [0.2141, 0.2232] | 0.642 [0.631, 0.653] |  | -0.0055 [-0.0109, -0.0003] |
| LR: rank + Elo | 0.6208 [0.6106, 0.6314] | 0.2164 [0.2120, 0.2211] | 0.646 [0.635, 0.657] | -0.0051 [-0.0065, -0.0035] | -0.0106 [-0.0146, -0.0068] |
| LR: all features, no pace | 0.6097 [0.5985, 0.6210] | 0.2117 [0.2070, 0.2166] | 0.660 [0.647, 0.672] | -0.0161 [-0.0195, -0.0124] | -0.0216 [-0.0266, -0.0167] |
| LR: all features | 0.6095 [0.5979, 0.6210] | 0.2116 [0.2068, 0.2165] | 0.661 [0.649, 0.674] | -0.0164 [-0.0199, -0.0126] | -0.0219 [-0.0270, -0.0169] |
| GBM: all features | 0.6081 [0.5966, 0.6195] | 0.2110 [0.2062, 0.2159] | 0.657 [0.645, 0.669] | -0.0177 [-0.0216, -0.0139] | -0.0233 [-0.0283, -0.0181] |
| GBM: all features, recalibrated | 0.6069 [0.5961, 0.6175] | 0.2105 [0.2059, 0.2152] | 0.657 [0.645, 0.669] | -0.0190 [-0.0225, -0.0153] | -0.0245 [-0.0292, -0.0198] |
| Average of LR and GBM | 0.6066 [0.5956, 0.6175] | 0.2105 [0.2058, 0.2152] | 0.659 [0.647, 0.671] | -0.0192 [-0.0226, -0.0158] | -0.0247 [-0.0294, -0.0200] |

Expected calibration error: Rank baseline 0.018, Elo (tuned) 0.029, LR: rank + Elo 0.017, LR: all features, no pace 0.022, LR: all features 0.020, GBM: all features 0.026, GBM: all features, recalibrated 0.013, Average of LR and GBM 0.016

### By year (log loss)

| Year | n | Rank baseline | Elo (tuned) | LR: all features | GBM: all features, recalibrated |
|---|---|---|---|---|---|
| 2023 | 2698 | 0.6323 | 0.6271 | 0.6076 | 0.6078 |
| 2024 | 2803 | 0.6318 | 0.6183 | 0.6018 | 0.5970 |
| 2025 | 2565 | 0.6330 | 0.6289 | 0.6171 | 0.6159 |
| 2026 | 2346 | 0.6280 | 0.6300 | 0.6125 | 0.6076 |

### Error slices (GBM recalibrated)

| Slice | n | Accuracy | Log loss | Elo log loss | Rank log loss | Model − Elo |
|---|---|---|---|---|---|---|
| surface = Clay | 3314 | 0.656 | 0.6090 | 0.6341 | 0.6327 | -0.0251 |
| surface = Grass | 1237 | 0.669 | 0.5977 | 0.6154 | 0.6329 | -0.0177 |
| surface = Hard | 5861 | 0.655 | 0.6076 | 0.6234 | 0.6303 | -0.0158 |
| tourney_level = 250 | 3503 | 0.627 | 0.6401 | 0.6588 | 0.6657 | -0.0186 |
| tourney_level = 500 | 1802 | 0.668 | 0.5958 | 0.6091 | 0.6098 | -0.0133 |
| tourney_level = A | 69 | 0.551 | 0.6786 | 0.6963 | 0.6681 | -0.0177 |
| tourney_level = F | 45 | 0.756 | 0.5202 | 0.5002 | 0.5593 | +0.0200 |
| tourney_level = G | 2021 | 0.733 | 0.5313 | 0.5590 | 0.5702 | -0.0277 |
| tourney_level = M | 2909 | 0.633 | 0.6287 | 0.6459 | 0.6486 | -0.0172 |
| tourney_level = O | 63 | 0.762 | 0.4730 | 0.5040 | 0.5166 | -0.0310 |
| round = F | 242 | 0.620 | 0.6224 | 0.6396 | 0.6517 | -0.0171 |
| round = QF | 939 | 0.644 | 0.6174 | 0.6214 | 0.6413 | -0.0040 |
| round = R128 | 1792 | 0.654 | 0.6031 | 0.6346 | 0.6384 | -0.0315 |
| round = R16 | 1882 | 0.666 | 0.6026 | 0.6195 | 0.6254 | -0.0169 |
| round = R32 | 3323 | 0.661 | 0.6085 | 0.6278 | 0.6288 | -0.0193 |
| round = R64 | 1705 | 0.662 | 0.5999 | 0.6213 | 0.6235 | -0.0214 |
| round = RR | 48 | 0.688 | 0.5847 | 0.5756 | 0.5790 | +0.0091 |
| round = SF | 480 | 0.629 | 0.6246 | 0.6277 | 0.6499 | -0.0031 |
| best_of = 3.0 | 8376 | 0.639 | 0.6249 | 0.6418 | 0.6461 | -0.0169 |
| best_of = 5.0 | 2036 | 0.731 | 0.5327 | 0.5603 | 0.5708 | -0.0275 |
| rank gap = <0.3 | 1781 | 0.576 | 0.6681 | 0.6833 | 0.6935 | -0.0152 |
| rank gap = 0.3-0.7 | 2266 | 0.571 | 0.6711 | 0.6892 | 0.6887 | -0.0181 |
| rank gap = 0.7-1.2 | 2210 | 0.639 | 0.6410 | 0.6623 | 0.6669 | -0.0213 |
| rank gap = >1.2 | 4136 | 0.748 | 0.5274 | 0.5467 | 0.5539 | -0.0193 |
| min Elo matches = <20 | 3209 | 0.664 | 0.6098 | 0.6400 | 0.6298 | -0.0302 |
| min Elo matches = 20-100 | 4350 | 0.645 | 0.6138 | 0.6325 | 0.6402 | -0.0187 |
| min Elo matches = >100 | 2853 | 0.667 | 0.5930 | 0.5998 | 0.6197 | -0.0068 |
| pace = slow | 3130 | 0.670 | 0.5963 | 0.6172 | 0.6205 | -0.0210 |
| pace = medium | 5596 | 0.645 | 0.6160 | 0.6345 | 0.6397 | -0.0185 |
| pace = fast | 1686 | 0.671 | 0.5962 | 0.6131 | 0.6240 | -0.0170 |
