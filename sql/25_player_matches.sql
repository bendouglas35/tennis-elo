-- Long format: one row per player per rateable match, with that player's own serve stats and
-- return stats derived by inverting the opponent's serve stats (the source has no return columns).
-- Foundation for the lagged rolling features in v0.2. These columns are in-match outcomes:
-- only ever use them through windows that end strictly BEFORE the match being predicted.

create or replace table player_matches as
select
    match_id, tourney_id, tourney_date, match_num, surface, tourney_level, round, best_of, minutes,
    winner_id                         as player_id,
    winner_name                       as player_name,
    loser_id                          as opponent_id,
    true                              as won,
    winner_rank                       as rank,
    winner_rank_points                as rank_points,
    loser_rank                        as opp_rank,
    w_ace                             as ace,
    w_df                              as df,
    w_svpt                            as svpt,
    w_1stWon + w_2ndWon               as sv_pts_won,
    w_bpSaved                         as bp_saved,
    w_bpFaced                         as bp_faced,
    l_svpt                            as ret_pts_played,
    l_svpt - l_1stWon - l_2ndWon      as ret_pts_won
from matches
where is_rateable
union all
select
    match_id, tourney_id, tourney_date, match_num, surface, tourney_level, round, best_of, minutes,
    loser_id, loser_name, winner_id, false,
    loser_rank, loser_rank_points, winner_rank,
    l_ace, l_df, l_svpt, l_1stWon + l_2ndWon, l_bpSaved, l_bpFaced,
    w_svpt, w_svpt - w_1stWon - w_2ndWon
from matches
where is_rateable;
