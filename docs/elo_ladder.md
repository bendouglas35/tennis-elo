# v0.3: Elo ladder

Code: `src/tennis/elo.py` (engine), `src/tennis/ladder.py` (tuning and report), tests in
`tests/test_elo.py`. Run with `make elo`.

## Protocol

- Ratings run sequentially from 2000 over every rateable match, in tournament-week and round order.
  Matches that share an order key are simultaneous: every pre-match rating in the group is read
  before any update is applied. Team events and the Next Gen Finals never update ratings and are
  never scored, but they reset a player's inactivity clock.
- Each extension is added one at a time. Its parameters are tuned on **2008-2021 only** (2000-2007
  is rating warm-up) by log loss, and a step is **kept only if it lowers the tuning-window log loss**.
  A step that does not help is still reported, but the next step builds on the previous kept
  configuration.
- The validation year (2022) is only scored, never used to choose a value. Intervals are 95%
  percentile intervals from a bootstrap that resamples whole tournaments (2,000 draws), and every
  difference is paired on the same draws. Population: standard tour matches with both ranks known
  (n = 2,625 of 2,630).
- Probabilities are the raw Elo win probabilities (the 400-point scale was tuned, not recalibrated).

## Result

| Model | Val log loss (95% CI) | Val Brier | Val accuracy | Δ log loss vs previous kept | Δ vs rank baseline | Kept |
|---|---|---|---|---|---|---|
| Rank baseline | 0.6314 [0.6134, 0.6516] | 0.2205 [0.2126, 0.2293] | 0.641 [0.616, 0.663] |  |  |  |
| Elo, fixed K | 0.6205 [0.5995, 0.6448] | 0.2160 [0.2068, 0.2267] | 0.642 [0.614, 0.667] | -0.0108 [-0.0202, -0.0002] | -0.0108 [-0.0202, -0.0002] | yes |
| + dynamic K | 0.6189 [0.5977, 0.6435] | 0.2153 [0.2061, 0.2258] | 0.647 [0.619, 0.670] | -0.0017 [-0.0044, 0.0011] | -0.0125 [-0.0226, -0.0014] | no |
| + rating scale | 0.6205 [0.5995, 0.6448] | 0.2160 [0.2068, 0.2267] | 0.642 [0.614, 0.667] | 0.0000 [0.0000, 0.0000] | -0.0108 [-0.0202, -0.0002] | no |
| + best-of-5 conversion | 0.6181 [0.5965, 0.6426] | 0.2152 [0.2060, 0.2258] | 0.643 [0.614, 0.667] | -0.0025 [-0.0038, -0.0012] | -0.0133 [-0.0229, -0.0026] | yes |
| + surface blend | 0.6164 [0.5966, 0.6383] | 0.2145 [0.2058, 0.2241] | 0.644 [0.616, 0.668] | -0.0017 [-0.0074, 0.0035] | -0.0150 [-0.0244, -0.0052] | yes |
| + layoff uncertainty | 0.6156 [0.5958, 0.6374] | 0.2141 [0.2054, 0.2238] | 0.647 [0.619, 0.671] | -0.0007 [-0.0013, -0.0001] | -0.0157 [-0.0250, -0.0060] | yes |
| + margin of victory | 0.6137 [0.5928, 0.6368] | 0.2131 [0.2041, 0.2234] | 0.653 [0.625, 0.678] | -0.0020 [-0.0037, -0.0000] | -0.0177 [-0.0273, -0.0072] | yes |
| + retirement scaling | 0.6136 [0.5928, 0.6368] | 0.2131 [0.2040, 0.2234] | 0.654 [0.626, 0.679] | -0.0001 [-0.0007, 0.0006] | -0.0178 [-0.0276, -0.0073] | no |

Final configuration (stored as `elo.TUNED`): fixed K = 24, best-of-5 conversion, surface blend 50/50,
layoff handling (grace 60 days, time constant 180 days, K boost x1 at full decay), margin-of-victory
scaling 1.0. 

## What the ladder says

- **Elo beats the ranking baseline**, by about 0.018 log loss at the end of the ladder
  (interval excludes zero), but each single extension is small and most of them are within noise
  on one validation year. The two clearest steps are the best-of-5 conversion and the move from the
  rank baseline to a rating itself.
- **Dynamic K does not beat a tuned fixed K** on the training window (0.6011 vs 0.6008), even with a
  floor on K. On the validation year it looks slightly better (-0.0017, interval spans zero), but the
  rule was fixed in advance, so it is not kept. With 25 years of data per player, a fixed K of
  24 is about as good as a decaying one.
- **Retirement scaling does nothing measurable**: retirements are about 3% of matches, and halving their updates changes the validation log loss by -0.0001 (interval -0.0007 to +0.0006); on the training window it is slightly worse, so it is not kept.
- **The "rating scale" step finds nothing**: 400 is already the best of {350, 400, 450}.
- **Layoff handling** is the one extension that interacts with the K rule: the effective-match decay
  has no effect with a fixed K, so the kept version is the K boost.
