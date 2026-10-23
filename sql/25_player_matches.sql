-- Long format: one row per player per rateable match, with that player's own serve stats and
-- return stats derived by inverting the opponent's serve stats (the source has no return columns).
-- Foundation for the lagged rolling features (features.py). These columns are in-match outcomes:
-- only ever use them through windows that end strictly BEFORE the match being predicted.

create or replace table player_matches as
select
    match_id, tourney_id, event_id, tourney_date, ord_num, match_num, surface, indoor_filled,
    tourney_level, round, best_of, minutes, is_team_event,
    coalesce(sets_w, 0) + coalesce(sets_l, 0)                 as sets_played,
    winner_id                         as player_id,
    winner_name                       as player_name,
    loser_id                          as opponent_id,
    true                              as won,
    false                             as retired,       -- the retiring player is the match loser
    winner_rank                       as rank,
    winner_rank_points                as rank_points,
    winner_seed                       as seed,
    winner_entry                      as entry,
    games_w                           as games_won,
    games_w + games_l                 as games_played,
    coalesce(w_svpt > 0 and l_svpt > 0, false)  as has_stats,
    w_ace                             as ace,
    w_df                              as df,
    w_svpt                            as svpt,
    w_1stIn                           as first_in,
    w_1stWon                          as first_won,
    w_2ndWon                          as second_won,
    w_1stWon + w_2ndWon               as sv_pts_won,
    w_bpSaved                         as bp_saved,
    w_bpFaced                         as bp_faced,
    l_svpt                            as ret_pts_played,
    l_svpt - l_1stWon - l_2ndWon      as ret_pts_won,
    l_bpFaced - l_bpSaved             as bp_won,
    l_bpFaced                         as bp_chances
from matches
where is_rateable
union all
select
    match_id, tourney_id, event_id, tourney_date, ord_num, match_num, surface, indoor_filled,
    tourney_level, round, best_of, minutes, is_team_event,
    coalesce(sets_w, 0) + coalesce(sets_l, 0),
    loser_id, loser_name, winner_id, false,
    coalesce(score_status = 'retired', false),
    loser_rank, loser_rank_points, loser_seed, loser_entry,
    games_l, games_w + games_l,
    coalesce(w_svpt > 0 and l_svpt > 0, false),
    l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_1stWon + l_2ndWon, l_bpSaved, l_bpFaced,
    w_svpt, w_svpt - w_1stWon - w_2ndWon,
    w_bpFaced - w_bpSaved, w_bpFaced
from matches
where is_rateable;
