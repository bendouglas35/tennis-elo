"""Court pace: how much a court favours the server, beyond who is serving.

Pipeline (all SQL, all leakage-safe):

1. `pace_match`: for every standard tour match, compare the aces and serve points won that actually
   happened with what the two players' own *lagged* records predict (`form_overall`, which only sees
   earlier matches), and combine the two standardised residuals into one composite.
       expected aces      = serve points x server ace rate x returner aces-allowed rate / tour ace rate
       expected pts won   = serve points x (server serve-won + returner allowed - tour serve-won)
   The tour rates are cumulative over strictly earlier tournament weeks.
2. `pace_edition`: one row per tournament edition (event id, surface, indoor, year) with its match
   count and composite sum.
3. `pace_static`: pooled over the previous `editions` editions of the same (event, surface, indoor)
   from earlier calendar years only, with recency weight rho ** (years of age beyond the first), shrunk
   to the surface/indoor mean over all earlier years with `m_static` pseudo-matches.
4. `match_pace`: running pace = earlier rounds of the same edition (strictly earlier ord_num), shrunk
   to the static pace with `m_running` pseudo-matches, plus a sparse altitude lookup.

A match's own in-match stats never reach its own pace: its residual is only used by later matches
(later rounds of the same edition, later years of the same event, and the tour/surface means).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaceParams:
    # Defaults tuned on training years only (2004-2021) with `pace_tune`; see docs/feature_plan.md.
    rho: float = 0.6          # recency weight per year of age beyond the first
    editions: int = 5         # most previous editions pooled
    m_static: float = 25.0    # pseudo-matches shrinking static pace to the surface/indoor mean
    m_running: float = 10.0   # pseudo-matches shrinking running pace to the static pace


# Standard deviations of the per-match residuals (rate per serve point), measured on training
# matches only (year <= 2021) with `scales()`. Fixed constants, so that deleting later data (the
# truncation check) cannot change them.
SD_ACE = 0.0331
SD_SP = 0.0473

# Sparse hand-built lookup: events held at roughly 500 m or more above sea level (rounded city
# elevations in metres). Everything else is 0. Keyed by the stable event id (suffix of tourney_id).
ALTITUDE_M = {
    "359": 2640,   # Bogota (2000-01)
    "6718": 2640,  # Bogota (2013-15)
    "7161": 2850,  # Quito
    "807": 2240,   # Mexico City
    "5012": 1750,  # Johannesburg
    "314": 1050,   # Gstaad
    "319": 760,    # Kitzbuhel
    "1536": 650,   # Madrid Masters
    "8996": 520,   # Santiago
}


def tour_mean_sql() -> str:
    return """
create or replace table pace_tour_mean as
with d as (
    select tourney_date, sum(ace)::double as ace, sum(svpt)::double as svpt,
           sum(sv_pts_won)::double as svw
    from player_matches where has_stats and not is_team_event group by 1
)
select tourney_date,
       sum(ace) over w / nullif(sum(svpt) over w, 0) as mean_ace,
       sum(svw) over w / nullif(sum(svpt) over w, 0) as mean_sv
from d
window w as (order by tourney_date range between unbounded preceding and interval 1 day preceding)
"""


def match_sql(sd_ace: float = SD_ACE, sd_sp: float = SD_SP) -> str:
    return f"""
create or replace table pace_match as
with pm as (
    select p.match_id, p.tourney_id, p.tourney_date, p.won, p.ace, p.svpt, p.sv_pts_won, p.has_stats,
           f.f_ace_rate, f.f_ace_allowed_rate, f.f_sv_won_pct, f.f_ret_won_pct
    from player_matches p join form_overall f using (match_id, player_id)
), j as (
    select a.match_id, a.tourney_id, a.tourney_date,
           (a.ace + b.ace)::double as aces, (a.sv_pts_won + b.sv_pts_won)::double as sw,
           (a.svpt + b.svpt)::double as svpt,
           a.svpt * a.f_ace_rate * b.f_ace_allowed_rate / t.mean_ace
             + b.svpt * b.f_ace_rate * a.f_ace_allowed_rate / t.mean_ace as exp_aces,
           a.svpt * (a.f_sv_won_pct + 1 - b.f_ret_won_pct - t.mean_sv)
             + b.svpt * (b.f_sv_won_pct + 1 - a.f_ret_won_pct - t.mean_sv) as exp_sw
    from pm a
    join pm b on a.match_id = b.match_id and a.won and not b.won
    join pace_tour_mean t on t.tourney_date = a.tourney_date
    where a.has_stats and a.svpt > 0 and b.svpt > 0
      and a.f_ace_rate is not null and b.f_ace_rate is not null
      and a.f_ace_allowed_rate is not null and b.f_ace_allowed_rate is not null
      and a.f_sv_won_pct is not null and b.f_sv_won_pct is not null
      and a.f_ret_won_pct is not null and b.f_ret_won_pct is not null
      and t.mean_ace > 0 and t.mean_sv > 0
)
select j.match_id, j.tourney_id, year(j.tourney_date) as yr,
       (aces - exp_aces) / svpt as ace_res, (sw - exp_sw) / svpt as sp_res,
       0.5 * ((aces - exp_aces) / svpt / {sd_ace!r} + (sw - exp_sw) / svpt / {sd_sp!r}) as comp
from j join matches m on m.match_id = j.match_id
where m.is_rateable and not m.is_team_event and not m.is_nonstandard_format
"""


def edition_sql() -> str:
    return """
create or replace table pace_keys as
select tourney_id, any_value(event_id) as event_id, any_value(surface) as surface,
       coalesce(any_value(indoor_filled), 'U') as ind, min(tourney_date) as tourney_date,
       year(min(tourney_date)) as yr
from matches where is_rateable group by tourney_id;

create or replace table pace_edition as
select k.*, count(pm.comp) as n, coalesce(sum(pm.comp), 0.0) as s
from pace_keys k left join pace_match pm using (tourney_id)
group by all
"""


def static_sql(p: PaceParams) -> str:
    return f"""
create or replace table pace_static as
with pairs as (
    select e.tourney_id, p.n, p.s,
           pow({p.rho!r}, greatest((e.yr - p.yr) - 1, 0)) as w,
           row_number() over (partition by e.tourney_id order by p.tourney_date desc) as rk
    from pace_edition e
    join pace_edition p on p.event_id = e.event_id and p.surface is not distinct from e.surface
                       and p.ind = e.ind and p.yr < e.yr
), pooled as (
    select tourney_id, sum(w * s) as ws, sum(w * n) as wn, count(*) as n_prior
    from pairs where rk <= {p.editions} group by tourney_id
), mu as (
    select e.tourney_id, sum(p.s) / nullif(sum(p.n), 0) as mu0
    from pace_edition e
    join pace_edition p on p.surface is not distinct from e.surface and p.ind = e.ind and p.yr < e.yr
    group by e.tourney_id
)
select k.tourney_id,
       (coalesce(po.ws, 0) + {p.m_static!r} * coalesce(mu.mu0, 0))
         / (coalesce(po.wn, 0) + {p.m_static!r}) as pace_static,
       coalesce(po.n_prior, 0) as pace_n_prior
from pace_keys k
left join pooled po using (tourney_id)
left join mu using (tourney_id)
"""


def altitude_sql() -> str:
    rows = ", ".join(f"('{k}', {v})" for k, v in ALTITUDE_M.items())
    return f"create or replace table altitude as select * from (values {rows}) t(event_id, altitude_m)"


def final_sql(p: PaceParams) -> str:
    return f"""
create or replace table match_pace as
with base as (
    select m.match_id, m.tourney_id, m.event_id, m.ord_num, pm.comp
    from matches m left join pace_match pm using (match_id)
    where m.is_rateable
), run as (
    select match_id, tourney_id, event_id,
           coalesce(sum(comp) over w, 0.0) as rs, count(comp) over w as rn
    from base
    window w as (partition by tourney_id order by ord_num
                 range between unbounded preceding and 1 preceding)
)
select r.match_id, st.pace_static,
       (r.rs + {p.m_running!r} * st.pace_static) / (r.rn + {p.m_running!r}) as pace_running,
       r.rn as pace_n_running, st.pace_n_prior,
       coalesce(al.altitude_m, 0) / 1000.0 as altitude_km
from run r
join pace_static st using (tourney_id)
left join altitude al on al.event_id = r.event_id
"""


def build_pace(con, params: PaceParams | None = None) -> None:
    """Create pace_match, pace_edition, pace_static, altitude and match_pace.

    Needs `matches`, `player_matches` and `form_overall` (so call it after the form tables).
    """
    p = params or PaceParams()
    con.execute(tour_mean_sql())
    con.execute(match_sql())
    con.execute(edition_sql())
    con.execute(static_sql(p))
    con.execute(altitude_sql())
    con.execute(final_sql(p))


def scales(con, train_end_year: int) -> tuple[float, float]:
    """SD of the raw residuals on training matches; used once to set SD_ACE and SD_SP."""
    return con.execute(
        "select stddev_samp(ace_res), stddev_samp(sp_res) from pace_match where yr <= ?",
        [train_end_year]).fetchone()
