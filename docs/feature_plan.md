# Feature plan (v0.2)

Status: agreed scope, not yet built. Items marked **(default)** are my assumptions, open to change.

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
