-- Typed, cleaned match table. One row per match, winner/loser orientation (as in the source).
-- Inputs (created by build.py): raw_matches (all varchar), score_parsed (one row per distinct score).
-- NOTE: columns minutes .. l_bpFaced describe what happened INSIDE the match. They must never be
-- used as features for that same match; only lagged aggregates are allowed (see sql/30_*).

create or replace table matches as
select
    m.tourney_id || '-' || m.match_num                    as match_id,
    m.tourney_id,
    m.tourney_name,
    m.surface,
    m.indoor,                                                         -- raw value; null for Sackmann data
    m.tourney_level,
    try_strptime(m.tourney_date, '%Y%m%d')::date           as tourney_date,   -- Monday of tournament week
    try_cast(m.match_num as integer)                       as match_num,
    m.round,
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

    m.loser_id,
    m.loser_name,
    m.loser_hand,
    try_cast(m.loser_ht as integer)                        as loser_ht,
    m.loser_ioc,
    try_cast(m.loser_age as double)                        as loser_age,
    try_cast(m.loser_rank as integer)                      as loser_rank,
    try_cast(m.loser_rank_points as integer)               as loser_rank_points,

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
    -- Davis Cup ties: different incentives, excluded later from core ratings
    (m.tourney_level = 'D')                                as is_team_event
from raw_matches m
left join score_parsed s on m.score is not distinct from s.score;
