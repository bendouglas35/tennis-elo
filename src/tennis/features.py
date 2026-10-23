"""Leakage-safe, pre-match features built in DuckDB with window functions.

Every window ends at "1 PRECEDING" on `ord_num` (see sql/10_matches.sql), so a match only ever sees
strictly earlier matches; matches with an equal key are treated as simultaneous. Form features are
exponentially weighted: a match k stats-matches ago has weight 0.5 ** (k / half_life). The weighted
sums are computed with cumulative window sums scaled by d**(-c), where c counts earlier matches:
ew_t = d**(c_t - 1) * sum_j d**(-c_j) * x_j. That is exact, set-based, and avoids recursion.
"""
from __future__ import annotations

# name, numerator, denominator, gate: "stats" decays/updates only on matches with serve stats,
# "all" on every rateable match. Ratios are point-weighted: sum(w*num) / sum(w*den).
STATS = [
    ("first_in_pct", "first_in", "svpt", "stats"),
    ("first_won_pct", "first_won", "first_in", "stats"),
    ("second_won_pct", "second_won", "(svpt - first_in)", "stats"),
    ("ace_rate", "ace", "svpt", "stats"),
    ("df_rate", "df", "svpt", "stats"),
    ("sv_won_pct", "sv_pts_won", "svpt", "stats"),
    ("bp_saved_pct", "bp_saved", "bp_faced", "stats"),
    ("ret_won_pct", "ret_pts_won", "ret_pts_played", "stats"),
    ("bp_conv_pct", "bp_won", "bp_chances", "stats"),
    ("win_rate", "won::int", "1", "all"),
    ("game_win_pct", "games_won", "games_played", "all"),
]
SURFACE_STATS = ["sv_won_pct", "ret_won_pct", "ace_rate", "win_rate", "game_win_pct"]

FATIGUE_COLS = [
    "days_since_last", "career_matches", "tourney_matches_before", "tourney_minutes_before",
    "tourney_sets_before", "n_14d", "minutes_14d", "n_28d", "surface_matches_365d",
]
DIFF_FEATURES = (
    [f"f_{s[0]}" for s in STATS] + ["f_dominance_ratio", "f_eff_stats"]
    + [f"fs_{s}" for s in SURFACE_STATS] + ["fs_dominance_ratio", "fs_eff_stats"]
    + FATIGUE_COLS
)
RAW_FEATURES = ["f_n_stats", "prev_retired", "is_seeded", "is_qualifier", "is_wildcard", "is_lucky_loser"]

_UNBOUNDED = "range between unbounded preceding and 1 preceding"


def range_frame(days: int) -> str:
    """RANGE frame over ord_num covering earlier matches with tourney_date within days-1 days.

    ord_num = (base * 1024 + sub), base = date_days * 16 + round_rank. A prior match k days earlier
    has a gap of at least 1024*(16k - 7) - 1023 and at most 1024*(16k + 7) + 1023, so the offset
    1024 * (16 * days - 8) includes every k <= days - 1 and excludes every k >= days.
    """
    return f"range between {1024 * (16 * days - 8)} preceding and 1 preceding"


def form_sql(table: str, prefix: str, part: str, stats: list, half_life: float, min_eff: float) -> str:
    d = 0.5 ** (1.0 / half_life)
    win = f"partition by {part} order by ord_num {_UNBOUNDED}"
    nd = ", ".join(
        f"coalesce({num}, 0)::double as n_{s}, coalesce({den}, 0)::double as d_{s}"
        for s, num, den, _ in stats
    )
    sums = ", ".join(
        f"sum({'ws' if g == 'stats' else 'wa'} * n_{s}) over ({win}) as sn_{s}, "
        f"sum({'ws' if g == 'stats' else 'wa'} * d_{s}) over ({win}) as sd_{s}"
        for s, _, _, g in stats
    )
    outs = ", ".join(
        f"case when {'es' if g == 'stats' else 'ea'} >= {min_eff} and sd_{s} > 0 "
        f"then sn_{s} / sd_{s} end as {prefix}{s}"
        for s, _, _, g in stats
    )
    return f"""
create or replace table {table} as
with l1 as (
    select match_id, player_id, surface, ord_num, has_stats, {nd},
           count(*) filter (where has_stats) over ({win}) as cs,
           count(*) over ({win}) as ca
    from player_matches
), l2 as (
    select *, case when has_stats then pow({d!r}, -cs) else 0.0 end as ws,
              pow({d!r}, -ca) as wa
    from l1
), l3 as (
    select match_id, player_id, cs, ca, {sums},
           sum(ws) over ({win}) as sws, sum(wa) over ({win}) as swa
    from l2
), l4 as (
    select *, case when cs >= 1 then sws * pow({d!r}, cs - 1) else 0.0 end as es,
              case when ca >= 1 then swa * pow({d!r}, ca - 1) else 0.0 end as ea
    from l3
)
select match_id, player_id, cs as {prefix}n_stats, ca as {prefix}n_all,
       es as {prefix}eff_stats, ea as {prefix}eff_all, {outs}
from l4
"""


def fatigue_sql() -> str:
    prior = f"partition by player_id order by ord_num {_UNBOUNDED}"
    tour = f"partition by player_id, tourney_id order by ord_num {_UNBOUNDED}"
    w14 = f"partition by player_id order by ord_num {range_frame(14)}"
    w28 = f"partition by player_id order by ord_num {range_frame(28)}"
    wsurf = f"partition by player_id, surface order by ord_num {range_frame(365)}"
    return f"""
create or replace table fatigue as
select match_id, player_id,
    tourney_date - max(tourney_date) over ({prior})          as days_since_last,
    arg_max(retired::int, ord_num) over ({prior})            as prev_retired,
    count(*) over ({prior})                                  as career_matches,
    count(*) over ({tour})                                   as tourney_matches_before,
    coalesce(sum(minutes) over ({tour}), 0)                  as tourney_minutes_before,
    coalesce(sum(sets_played) over ({tour}), 0)              as tourney_sets_before,
    count(*) over ({w14})                                    as n_14d,
    coalesce(sum(minutes) over ({w14}), 0)                   as minutes_14d,
    count(minutes) over ({w14})                              as n_min_14d,
    count(*) over ({w28})                                    as n_28d,
    count(*) over ({wsurf})                                  as surface_matches_365d
from player_matches
"""


def player_features_sql() -> str:
    return """
create or replace table player_features as
select pm.match_id, pm.player_id, pm.rank, pm.rank_points,
       (pm.seed is not null)                       as is_seeded,
       coalesce(pm.entry = 'Q', false)             as is_qualifier,
       coalesce(pm.entry = 'WC', false)            as is_wildcard,
       coalesce(pm.entry = 'LL', false)            as is_lucky_loser,
       fo.* exclude (match_id, player_id),
       fs.* exclude (match_id, player_id),
       fa.* exclude (match_id, player_id),
       fo.f_ret_won_pct / nullif(1 - fo.f_sv_won_pct, 0)    as f_dominance_ratio,
       fs.fs_ret_won_pct / nullif(1 - fs.fs_sv_won_pct, 0)  as fs_dominance_ratio
from player_matches pm
join form_overall fo using (match_id, player_id)
join form_surface fs using (match_id, player_id)
join fatigue fa using (match_id, player_id)
"""


def model_features_sql() -> str:
    diffs = ", ".join(f"pa.{c} - pb.{c} as d_{c}" for c in DIFF_FEATURES)
    raws = ", ".join(f"pa.{c} as {c}_a, pb.{c} as {c}_b" for c in RAW_FEATURES)
    return f"""
create or replace table model_features as
select ms.match_id, ms.tourney_id, ms.event_id, ms.tourney_date, ms.ord_num,
       ms.surface, ms.indoor_filled, ms.tourney_level, ms.round, ms.best_of, ms.draw_size,
       ms.is_team_event, ms.is_nonstandard_format, ms.player_a_id, ms.player_b_id, ms.a_wins,
       case when ms.rank_a > 0 and ms.rank_b > 0 then ln(ms.rank_b::double / ms.rank_a) end
                                                                  as d_log_rank,
       case when ms.rank_points_a > 0 and ms.rank_points_b > 0
            then ln(ms.rank_points_a::double / ms.rank_points_b) end as d_log_rank_points,
       ms.age_a - ms.age_b                                         as d_age,
       ms.ht_a - ms.ht_b                                           as d_ht,
       coalesce(ms.hand_a = 'L', false)                            as lefty_a,
       coalesce(ms.hand_b = 'L', false)                            as lefty_b,
       {diffs}, {raws}
from match_sides ms
join player_features pa on pa.match_id = ms.match_id and pa.player_id = ms.player_a_id
join player_features pb on pb.match_id = ms.match_id and pb.player_id = ms.player_b_id
"""


def build_features(con, half_life: float = 20.0, min_eff: float = 3.0) -> None:
    """Create form_overall, form_surface, fatigue, player_features and model_features."""
    con.execute(form_sql("form_overall", "f_", "player_id", STATS, half_life, min_eff))
    surf = [s for s in STATS if s[0] in SURFACE_STATS]
    con.execute(form_sql("form_surface", "fs_", "player_id, surface", surf, half_life, min_eff))
    con.execute(fatigue_sql())
    con.execute(player_features_sql())
    con.execute(model_features_sql())
