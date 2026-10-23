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
