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
    home_wins: float
    home_draws: float
    home_loses: float
    home_goals_scored: float
    home_goals_conceded: float
    home_goal_difference: int
    home_last_5_matches: AggregatedMatchData
    home_last_5_matches_at_home: AggregatedMatchData
    away_position: int
    away_played_matches: int
    away_wins: float
    away_draws: float
    away_loses: float
    away_goals_scored: float
    away_goals_conceded: float
    away_goal_difference: int
    away_last_5_matches: AggregatedMatchData
    away_last_5_matches_at_away: AggregatedMatchData
    result: int
    home_odds: float
    draw_odds: float
    away_odds: float
    last_3_matches_between_teams: AggregatedMatchData
