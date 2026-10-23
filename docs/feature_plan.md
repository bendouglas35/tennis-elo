# Feature plan (v0.2)

Status: step 1 (form, fatigue, entry) and step 2 (court pace) are built and tested. Elo is next. Items marked
**(default)** are my assumptions, open to change.

## Ordering and leakage rules

- `tourney_date` is only the Monday of the tournament week. "Before this match" therefore means a
  strictly earlier **order key** = (`tourney_date`, `round_rank`), with
  `R128 < R64 < R32 < R16 < QF = RR < SF < F = BR`. Matches with the same key are simultaneous and
  never feed each other. This also covers the nine 2025 events that have no `match_num`.
- In-match columns (`minutes` .. `l_bpFaced`, score details) reach a feature only through windows
  ending strictly before the order key.
- Tests (v0.2 deliverable):
  1. **Truncation:** for a random sample of matches, rebuild features after deleting every match at
     or after its order key; the match's features must not change.
  2. **Same slot:** matches in the same tournament and round never influence each other.
  3. **No self-reference:** no feature column reads the target match's own in-match columns.

## Features (all modelled as A minus B differences, plus a few raw context columns)

| Group | Features |
|---|---|
| Already in data | log rank ratio, rank points, age, height, handedness matchup, surface, `indoor`, level (250/500/M/G/F), round, best-of, draw size |
| Entry | seeded, qualifier, wildcard, lucky loser (needs `seed`/`entry` carried into `matches`) |
| Serve form | first-serve %, first-serve and second-serve points won, ace %, double-fault %, break points saved % |
| Return form | return points won %, break-point conversion % (opponent's serve stats inverted) |
| Combined form | dominance ratio (overall and surface-specific), game-win %, tiebreak win %, match win % |
| Rest / fatigue | days since last match, minutes and sets earlier this tournament, matches in last 14 days, retired in previous match |
| Experience | career matches, matches on this surface in last 365 days |
| Elo (Python, v0.3) | overall and surface-blended rating, uncertainty / days inactive |
| Pace | static pace, running in-tournament pace, altitude |

- **Form windows:** exponentially weighted over previous matches **(default)**, half-life tuned on
  validation over {10, 20, 40} matches. Matches without serve stats are skipped, not zero-filled.
  Too few prior matches gives NULL plus a flag.
- **Indoor fill:** about 9% of matches have no `indoor` value. Fill from the same event and surface
  in other years where unambiguous, otherwise leave NULL.
- **Dropped:** weather, ball brand, session type, start times, travel, backhand type, prize money.
  **Stretch:** home-tournament flag (needs a hand-built host-country table for ~136 events).

## Court pace

Keyed by (event id, surface, indoor), because event ids are stable across sponsor renames but 15 of
136 events changed surface.

1. **Per-match residuals.** For each match, compare actual aces and serve-points-won with what the
   two players' own lagged records predict: expected ace rate = server's ace rate x returner's
   aces-allowed rate / tour mean; expected serve-points-won = server rate + returner-allowed rate
   minus tour mean. Player rates use earlier matches only.
2. **Composite:** standardise the two residual measures and average them **(chosen)**.
3. **Static pace:** edition-level composites pooled over the previous 3-5 editions with recency
   weights **(default: 0.6 per year)**, shrunk towards the surface/indoor mean
   **(default: 100 pseudo-matches)**. A match in year Y uses editions before Y only. Events with no
   prior edition (6 events, 274 matches from 2023 on) fall back to the surface/indoor mean.
4. **Running pace (chosen):** the same composite over earlier rounds of the current edition,
   shrunk heavily to the static value **(default: 30 pseudo-matches)**. In round 1 it equals the
   static pace. Meant to capture this year's real conditions (heat, new balls, resurfacing).
5. **Altitude:** sparse hand-built table of high-altitude events only (default 0 elsewhere),
   checked against public sources before use.
6. **Ablation:** surface only, then + indoor + altitude, then + static pace, then + running pace.
   Each step gets a bootstrap confidence interval; steps that do not help are reported as such.

## Population and warm-up decisions (defaults)

- **Team events** (Davis/Laver/ATP/United Cup) and Next Gen Finals: never a prediction target and
  never update Elo, but they do count towards players' rolling form and fatigue, since the match
  load is real.
- **Warm-up:** ratings and form run from 2000; model training starts in 2003, so early cold-start
  ratings are not learned from. Pace residuals likewise need a few years of player history.
- **Frozen split unchanged:** train <= 2021, validate 2022, test >= 2023.

## Step 1 as built (differences from the plan above)

- **Features in `model_features`:** 11 point-weighted exponentially weighted form stats (overall),
  5 surface-specific ones, dominance ratio (overall and surface), rest and fatigue
  (`days_since_last`, tournament minutes/sets/rounds so far, matches in the last 14 and 28 days,
  career matches, matches on the surface in the last 365 days, previous-match retirement), entry
  flags (seeded / qualifier / wildcard / lucky loser), log rank ratio, log rank-points ratio, age,
  height and handedness. Cross-player features are A minus B (`d_*`); a few raw per-side values are
  kept (`*_a`, `*_b`).
- **Tiebreak win % dropped.** The parsed score gives the number of tiebreaks played but not who
  won each one. Game-win % and match win % are in.
- **Half-life 20 matches and minimum effective sample 3 are defaults, not tuned.** Tuning happens
  on the 2022 validation year in v0.4. Form is NULL when fewer than 3 effective matches exist
  (about 10% of matches overall).
- **Same-key matches:** 104 groups where a player has several matches with an identical ordering
  key, almost all in team events (ATP Cup, Davis Cup, Laver Cup). They are treated as
  simultaneous, which is safe (no leakage) at the cost of ignoring the earlier tie's load.
- **Leakage tests:** `make leakage` (truncation and perturbation on the real database) and the
  pytest suite, which also mutation-tests the checks themselves by injecting a leak.

## Step 2 as built: court pace

Code: `src/tennis/pace.py` (SQL stages), `src/tennis/pace_tune.py` (tuning), tests in `tests/test_pace.py`.
Columns added to `model_features`: `pace_static`, `pace_running`, `pace_n_running`, `pace_n_prior`,
`altitude_km`. They describe the match's court, so they are context columns, not A-minus-B differences.
Pace matters through interactions (a fast court helps big servers), which the v0.4 models must be
given explicitly (for example pace x difference in ace rate); a main effect is symmetric and cancels
out of the A/B target.

- **Residuals** follow the plan. One addition: a new form stat, `f_ace_allowed_rate` (aces hit against
  the player per return point), is needed for the expected ace rate. The tour rates in the
  expectation are cumulative over strictly earlier tournament weeks. 92% of standard tour matches get
  a residual (the rest lack serve stats or have a player with too little history).
- **Standardisation constants** (`SD_ACE = 0.0331`, `SD_SP = 0.0473`, the SD of the per-match residuals)
  were measured once on training years (<= 2021) and are fixed in code, so that deleting later data
  cannot move them.
- **Surface sanity check** (mean composite per match, whole sample): grass +0.31, carpet +0.34, indoor
  hard +0.08, outdoor hard -0.09, outdoor clay -0.25. Latest static pace puts Stuttgart, Halle,
  Queen's and Wimbledon at the fast end and Monte Carlo, Barcelona and Roland Garros at the slow end.
- **Persistence:** the correlation between an event's mean composite and the same event's the year
  before is 0.69 (1,443 pairs, editions with >= 20 matches).
- **Tuning (training years 2004-2021 only, scored by squared error against each edition's realised
  mean, weighted by matches):**

  | Static pace setting | Error |
  |---|---|
  | Surface/indoor mean only (no event history) | 0.1004 |
  | Pooled, rho 0.6, 5 editions, m = 100 (original default) | 0.0673 |
  | Pooled, rho 0.6, 5 editions, m = 25 (**chosen**) | 0.0604 |
  | Best of the grid (8 editions, m = 25) | 0.0602 |

  Using the event's own history cuts that error by about 40% relative to the surface mean alone, and the
  planned 100 pseudo-matches was too much shrinkage, so the default is now **25**. Five editions
  and rho 0.6 stay (eight editions is only 0.0002 better). Running pace, scored on the composite of
  each later-round match: no running pace 0.6072, m = 30 0.5838, **m = 10 0.5812** (chosen). The
  gain from running pace is small (about 4% of a noise-dominated error), which is expected: it only
  has earlier rounds of one week to learn from. The ablation tests whether it adds anything.
- **Fallbacks:** 7,406 matches have no previous edition and use the surface/indoor mean (386 of the
  11,509 matches from 2023 on; 2,703 of the 52,600 from 2004-2021).
- **Altitude:** nine events, all >= 500 m, as rounded city elevations from common public references
  (not venue-surveyed): Bogota 2,640 m, Quito 2,850 m, Mexico City 2,240 m, Johannesburg 1,750 m, Gstaad
  1,050 m, Kitzbuhel 760 m, Madrid 650 m, Santiago 520 m. Only 4,828 matches are affected; it is a
  weak candidate feature and the ablation will say whether it earns a place.
- **Ablation (surface -> + indoor/altitude -> + static pace -> + running pace)** is deferred to v0.4,
  since it needs the interaction features and the final model to be meaningful.

## Fixes made during step 2

- **Indoor coding:** the 2026 US Open (127 rows) is coded indoor = '0'; it is now normalised to 'O',
  otherwise it would have made that event look ambiguous for filling.
- **`prev_retired` tie-break (latent step 1 bug):** where a player has several matches with an
  identical order key (Davis Cup ties), the "previous match ended in retirement" flag was picked
  arbitrarily and changed with physical row order. The truncation check caught it. It is now
  deterministic: 1 if any of the latest simultaneous matches ended in retirement. Covered by a test.
- **Perturbation check redesigned:** pace and form legitimately share tour-wide and surface-wide
  means over earlier data, so perturbed matches from different weeks can affect each other's rows
  and the old "80 random late matches" design reported false leaks. It now perturbs matches from
  one week at a time (at most one per player and event), repeated for several weeks (8 on real data:
  31 matches, own rows 0 changed, 492,164 earlier-row comparisons 0 changed, 134,210 later rows
  changed). Fewer matches are perturbed per run than before, but each comparison is valid.
- **Leakage checks are mutation-tested for pace:** deliberately broken pace SQL (pooling the same
  year; running pace including the match itself) is caught.
