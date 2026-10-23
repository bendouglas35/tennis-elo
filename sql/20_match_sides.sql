-- Symmetric modelling table: one row per rateable match, players randomly labelled A / B.
--
-- If the winner were always "player 1", a model could learn "player 1 wins" and nothing else.
-- Orientation is a deterministic hash of match_id: reproducible, independent of the outcome,
-- and roughly 50/50. The target is a_wins.

create or replace table match_sides as
with oriented as (
    select *, (hash(match_id) % 2 = 0) as winner_is_a
    from matches
    where is_rateable
)
select
    match_id, tourney_id, event_id, tourney_name, tourney_date, ord_num, match_num, round, draw_size,
    surface, indoor, indoor_filled, tourney_level, best_of, is_team_event, is_nonstandard_format,
    case when winner_is_a then winner_id   else loser_id   end as player_a_id,
    case when winner_is_a then loser_id    else winner_id  end as player_b_id,
    case when winner_is_a then winner_name else loser_name end as player_a_name,
    case when winner_is_a then loser_name  else winner_name end as player_b_name,
    case when winner_is_a then winner_hand else loser_hand end as hand_a,
    case when winner_is_a then loser_hand  else winner_hand end as hand_b,
    case when winner_is_a then winner_ht   else loser_ht   end as ht_a,
    case when winner_is_a then loser_ht    else winner_ht  end as ht_b,
    case when winner_is_a then winner_age  else loser_age  end as age_a,
    case when winner_is_a then loser_age   else winner_age end as age_b,
    case when winner_is_a then winner_rank else loser_rank end as rank_a,
    case when winner_is_a then loser_rank  else winner_rank end as rank_b,
    case when winner_is_a then winner_rank_points else loser_rank_points end as rank_points_a,
    case when winner_is_a then loser_rank_points  else winner_rank_points end as rank_points_b,
    winner_is_a                                                 as a_wins
from oriented;
