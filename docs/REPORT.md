# Predicting ATP tennis matches: a leakage-safe Elo and gradient-boosting study

Ben Douglas · October 2026 · code: this repository

## Summary

I built a pipeline that predicts the winner of ATP tour matches before they are played, and measured
how much each ingredient (rankings, Elo, form and fatigue, court pace, more rating data, richer
features) actually adds. The emphasis is on validation: every feature is built from information
available before the match, the evaluation split was frozen in advance, and uncertainty is reported
with intervals that resample whole tournaments.

Headline results (log loss, lower is better; a coin flip is 0.693):

| Model | Validation 2022 | Frozen test 2023+ (n = 10,412) |
|---|---|---|
| Higher-ranked player (logistic on log rank ratio) | 0.632 | 0.631 [0.621, 0.642] |
| Tuned Elo | 0.613 | 0.626 [0.615, 0.637] |
| Logistic regression, all features | 0.596 | 0.610 [0.598, 0.621] |
| LightGBM, recalibrated | 0.597 | 0.607 [0.596, 0.618] |

The full model improves on rankings by about 0.025 log loss and on tuned Elo by about 0.019 (both
intervals exclude zero), and the gain holds in each of 2023-2026. Accuracy is about 66%. That is a
real but modest improvement. I make no claim about beating betting markets, whose closing prices
are very sharp, and I did not compare against them.

A later iteration (v0.5) added qualifying and Challenger results to the ratings, which lowered rolling
validation log loss by about 0.003 for both models. Three other ideas (rating trend, Glicko
uncertainty, a serve/return point model) did not help and were dropped.

## 1. Data and split

Ordinary ATP tour matches 2000-2026 (Davis Cup, Laver Cup, ATP Cup, United Cup and Next Gen Finals
excluded as targets; walkovers and defaults excluded). Training uses matches up to 2021, 2022 is the
validation year, and 2023 onwards is a test set touched at milestones only. The data snapshot is
pinned by file hash in `docs/data_snapshot.md`. For v0.5 I also used about 29,800 qualifying
matches (2007-2026) and about 121,900 Challenger matches (2000-2026) for ratings only.
Raw data is not redistributed.

## 2. Method

**SQL feature layer (DuckDB).** Rolling form, ace and serve/return rates, rest and fatigue (recent
minutes and matches), entry type, ranking, altitude, and court pace. Matches are ordered by
tournament week and round, and every window ends strictly before the match.

**Symmetric design.** Each match is labelled player A or B by a hash, and both models are exactly
antisymmetric: p(x) + p(mirror x) = 1 (tested). Logistic regression uses difference features with no
intercept; LightGBM is trained on the data plus its mirror image.

**Elo, built up one extension at a time** and tuned on 2008-2021 only: fixed K, best-of-5
conversion, surface blend, layoff uncertainty, margin of victory. Matches in the same tournament
round update ratings simultaneously, so no match sees another's result from the same round.

**Court pace.** Per-match ace and serve-points-won residuals against the players' own earlier
records, pooled over an event's previous editions. It explains most of the edition-level variation
that surface alone misses (error 0.100 to 0.060), but adds little to prediction (below).

**Models.** Logistic regression and LightGBM with expanding-window tuning on 2018-2021, then a
one-parameter calibration (slope 0.90).

**Uncertainty.** 2,000-draw bootstrap resampling whole tournaments; model comparisons are paired.

## 3. Leakage control

Three layers: (1) construction, with windows that end before the match; (2) a truncation test that
deletes all later data and checks no feature changes, and a perturbation test that overwrites
in-match statistics one week at a time and checks no feature changes; (3) tests that inject a
deliberate leak to prove these checks fail when they should. The frozen test years were scored once,
after the code and configuration were committed; the script refuses to run a second time.

## 4. Results

**Elo ladder (validation 2022).** Tuned Elo beats the rank baseline by 0.018 log loss. Margin of
victory, layoff uncertainty and the best-of-5 conversion each helped on the tuning years.

**Features.** Form, fatigue and rank information on top of Elo account for most of the further gain
(logistic regression on Elo plus rank scores 0.621 on the test set; all features 0.610).
The logistic regression matches the GBM within noise, which suggests the signal is close to linear
in these features. Court pace helps by about 0.0007 log loss, significant only for the logistic
regression.

**Calibration.** The recalibrated GBM has expected calibration error of 0.013; the Elo alone is
over-confident (0.029).

**Where the model helps most.** Best-of-5 matches, Grand Slams, early rounds, and players with few
Elo matches (-0.030 vs Elo). It helps least among well-established players (-0.007) and in
semi-finals and quarter-finals.

**v0.5: four attempted improvements** (rolling origin 2018-2022, n = 11,650, both models at fixed
v0.4 hyperparameters; a step is kept only if its interval excludes zero for both models):

| Step | LR Δ log loss | GBM Δ log loss | Decision |
|---|---|---|---|
| Elo learns from qualifying and Challenger results | -0.0033 [-0.0053, -0.0015] | -0.0034 [-0.0053, -0.0014] | keep |
| Rating trend (90 and 365 days) | +0.0001 | +0.0003 | drop |
| Glicko uncertainty | -0.0010 [-0.0016, -0.0005] | -0.0002 [-0.0010, +0.0006] | drop |
| Serve/return point model | -0.0002 | -0.0003 | drop |

Rank-seeding new players' ratings also gave no gain. Details: `docs/v05_improvements.md`.

## 5. Limitations

* The test set was scored once for the v0.4 model. The v0.5 model was chosen on development years
  and has **not** been scored on 2023+; a v0.5 test score would be a second look and would be
  reported as such.
* No comparison with bookmaker odds, so no claim about profitability.
* Gains are small: log loss differences of 0.001-0.03 correspond to changes of a few percentage
  points of accuracy at most.
* The 2026 season is partial, and one data source is used for the main matches.
* Elo gains from extra data show up in the ratings of players with little tour history, so they
  concentrate in a minority of matches.

## 6. Reproducing

`make setup data build leakage elo tune validate test` (see README). Raw CSVs are needed in
`data/raw` (tour matches) and, for v0.5, `data/raw_quali` and `data/raw_chall`; then
`python -m tennis.cli build-aux` and `python -m tennis.cli models v05`.

## 7. Next steps

Separate rating weights for qualifying versus Challenger results; calibration by match type;
rating players separately on serve and return; ITF-level results for very young players.
