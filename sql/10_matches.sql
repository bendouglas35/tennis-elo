-- Typed, cleaned match table. One row per match, winner/loser orientation (as in the source).
-- Inputs (created by build.py): raw_matches (all varchar), score_parsed (one row per distinct score).
-- NOTE: columns minutes .. l_bpFaced describe what happened INSIDE the match. They must never be
-- used as features for that same match; only lagged aggregates are allowed (see features.py).

create or replace table matches_core as
with rm as (
    -- TML codes outdoor as '0' for some 2026 rows (US Open); normalise to 'O'.
    select * replace (case when indoor = '0' then 'O' else indoor end as indoor) from raw_matches
), indoor_fill as (
    -- Indoor/outdoor is event metadata, known in advance (not outcome information). Fill a missing
    -- value only when the same event and surface is unambiguous in every other year.
    select split_part(tourney_id, '-', 2) as event_id, surface,
           case when count(distinct indoor) = 1 then any_value(indoor) end as fill
    from rm
    where indoor is not null
    group by 1, 2
)
select
    -- Some source tournaments (e.g. nine events in Aug-Oct 2025) have no match_num; fall back to
    -- tournament + round + player ids, which is unique within a tournament.
    m.tourney_id || '-' || coalesce(m.match_num, coalesce(m.round, '?') || '-' || m.winner_id || '-' || m.loser_id)
                                                           as match_id,
    m.tourney_id,
    split_part(m.tourney_id, '-', 2)                       as event_id,     -- stable across years/renames
    m.tourney_name,
    m.surface,
    m.indoor,                                                         -- raw value
    coalesce(m.indoor, f.fill)                             as indoor_filled,
    m.tourney_level,
    try_strptime(m.tourney_date, '%Y%m%d')::date           as tourney_date,   -- Monday of tournament week
    try_cast(m.match_num as integer)                       as match_num,
    m.round,
    try_cast(m.draw_size as integer)                       as draw_size,
    try_cast(m.best_of as integer)                         as best_of,
    try_cast(m.minutes as integer)                         as minutes,
    m.score,
    s.status                                               as score_status,
    s.sets_w, s.sets_l, s.games_w, s.games_l,
    s.tiebreaks, s.match_tiebreak, s.last_set_partial,

    m.winner_id,
    m.winner_name,
    m.winner_hand,
    try_cast(m.winner_ht as integer)                       as winner_ht,
    m.winner_ioc,
    try_cast(m.winner_age as double)                       as winner_age,
    try_cast(m.winner_rank as integer)                     as winner_rank,
    try_cast(m.winner_rank_points as integer)              as winner_rank_points,
    try_cast(m.winner_seed as integer)                     as winner_seed,
    upper(m.winner_entry)                                  as winner_entry,

    m.loser_id,
    m.loser_name,
    m.loser_hand,
    try_cast(m.loser_ht as integer)                        as loser_ht,
    m.loser_ioc,
    try_cast(m.loser_age as double)                        as loser_age,
    try_cast(m.loser_rank as integer)                      as loser_rank,
    try_cast(m.loser_rank_points as integer)               as loser_rank_points,
    try_cast(m.loser_seed as integer)                      as loser_seed,
    upper(m.loser_entry)                                   as loser_entry,

    try_cast(m.w_ace as integer)     as w_ace,     try_cast(m.l_ace as integer)     as l_ace,
    try_cast(m.w_df as integer)      as w_df,      try_cast(m.l_df as integer)      as l_df,
    try_cast(m.w_svpt as integer)    as w_svpt,    try_cast(m.l_svpt as integer)    as l_svpt,
    try_cast(m.w_1stIn as integer)   as w_1stIn,   try_cast(m.l_1stIn as integer)   as l_1stIn,
    try_cast(m.w_1stWon as integer)  as w_1stWon,  try_cast(m.l_1stWon as integer)  as l_1stWon,
    try_cast(m.w_2ndWon as integer)  as w_2ndWon,  try_cast(m.l_2ndWon as integer)  as l_2ndWon,
    try_cast(m.w_SvGms as integer)   as w_SvGms,   try_cast(m.l_SvGms as integer)   as l_SvGms,
    try_cast(m.w_bpSaved as integer) as w_bpSaved, try_cast(m.l_bpSaved as integer) as l_bpSaved,
    try_cast(m.w_bpFaced as integer) as w_bpFaced, try_cast(m.l_bpFaced as integer) as l_bpFaced,

    -- walkovers / defaults carry no (or little) information: excluded from ratings and training
    (s.status in ('completed', 'retired'))                 as is_rateable,
    -- Team events and exhibitions have different incentives: Davis Cup (level D) plus Laver Cup,
    -- ATP Cup and United Cup, which the source files under level A. Excluded later from core ratings.
    (m.tourney_level = 'D'
     or lower(m.tourney_name) like '%laver cup%'
     or lower(m.tourney_name) like '%atp cup%'
     or lower(m.tourney_name) like '%united cup%')         as is_team_event,
    -- Next Gen Finals use shortened sets, so scores/stats are not comparable to normal matches
    (lower(m.tourney_name) like '%next gen%')              as is_nonstandard_format
from rm m
left join score_parsed s on m.score is not distinct from s.score
left join indoor_fill f
       on split_part(m.tourney_id, '-', 2) = f.event_id and m.surface is not distinct from f.surface;

-- Ordering key. tourney_date is only the Monday of the tournament week, so "before this match"
-- is defined by ord_num = ((days * 16 + round_rank) * 1024 + sub):
--   round_rank orders rounds within a tournament; sub orders round-robin matches by match_num
--   (the only place one player has several matches in the same round). Equal ord_num means
--   "simultaneous": such matches never feed each other. Windows over ord_num in features.py use
--   RANGE frames ending at "1 PRECEDING", which excludes peers with an equal key.
create or replace table matches as
select c.*,
       case c.round when 'R128' then 1 when 'R64' then 2 when 'R32' then 3 when 'R16' then 4
                    when 'QF' then 5 when 'RR' then 5 when 'SF' then 6
                    when 'F' then 7 when 'BR' then 7 when '3rd/4th' then 7 else 4 end   as round_rank,
       (((c.tourney_date - date '1970-01-01') * 16
         + case c.round when 'R128' then 1 when 'R64' then 2 when 'R32' then 3 when 'R16' then 4
                        when 'QF' then 5 when 'RR' then 5 when 'SF' then 6
                        when 'F' then 7 when 'BR' then 7 when '3rd/4th' then 7 else 4 end) * 1024
        + case when c.round = 'RR' then least(coalesce(c.match_num, 0), 1023) else 0 end)::bigint
                                                                                      as ord_num
from matches_core c;

drop table matches_core;
