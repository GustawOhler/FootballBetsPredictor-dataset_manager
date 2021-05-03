from dataclasses import dataclass
from enum import Enum


class DatasetType(Enum):
    TRAIN = 'train'
    VAL = 'val'


@dataclass
class AggregatedMatchData:
    wins: float = 0.0
    draws: float = 0.0
    loses: float = 0.0
    scored_goals: float = 0.0
    conceded_goals: float = 0.0
    shots_fired: float = 0.0
    shots_fired_on_target: float = 0.0
    shots_conceded: float = 0.0
    shots_conceded_on_target: float = 0.0


@dataclass
class NNDatasetRow:
    home_position: int
    home_played_matches: int
    home_wins: int
    home_draws: int
    home_loses: int
    home_goals_scored: int
    home_goals_conceded: int
    home_goal_difference: int
    home_team_wins_in_last_5_matches: int
    home_team_draws_in_last_5_matches: int
    home_team_loses_in_last_5_matches: int
    home_team_scored_goals_in_last_5_matches: int
    home_team_conceded_goals_in_last_5_matches: int
    away_position: int
    away_played_matches: int
    away_wins: int
    away_draws: int
    away_loses: int
    away_goals_scored: int
    away_goals_conceded: int
    away_goal_difference: int
    away_team_wins_in_last_5_matches: int
    away_team_draws_in_last_5_matches: int
    away_team_loses_in_last_5_matches: int
    away_team_scored_goals_in_last_5_matches: int
    away_team_conceded_goals_in_last_5_matches: int
    result: int
    home_odds: float
    draw_odds: float
    away_odds: float
