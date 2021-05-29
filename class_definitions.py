from dataclasses import dataclass
from enum import Enum
from typing import List

results_dict = {'H': 0, 'D': 1, 'A': 2}

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
class SingleMatchForRootData:
    root_result: int = 0
    is_home: int = 0
    days_since_match: int = 0
    league_level: int = 0
    position: int = 0
    opposite_team_position: int = 0
    scored_goals: int = 0
    conceded_goals: int = 0
    shots_fired: int = 0
    shots_fired_on_target: int = 0
    shots_conceded: int = 0
    shots_conceded_on_target: int = 0
    corners_taken: int = 0
    opposite_corners: int = 0
    fouls_commited: int = 0
    opposite_fouls: int = 0
    cards: int = 0
    opposite_cards: int = 0


@dataclass
class DatasetWithSeparatedMatchesRow:
    match_id: int
    home_position: int
    home_played_matches: int
    home_wins: float
    home_draws: float
    home_loses: float
    home_goals_scored: float
    home_goals_conceded: float
    home_goal_difference: int
    home_last_4_matches: List[SingleMatchForRootData]
    away_position: int
    away_played_matches: int
    away_wins: float
    away_draws: float
    away_loses: float
    away_goals_scored: float
    away_goals_conceded: float
    away_goal_difference: int
    away_last_4_matches: List[SingleMatchForRootData]
    result: int
    home_odds: float
    draw_odds: float
    away_odds: float
    last_3_matches_between_teams: AggregatedMatchData
    home_last_5_matches_as_home: AggregatedMatchData
    away_last_5_matches_as_away: AggregatedMatchData
    home_position_last_season: int
    home_league_level_last_season: int
    away_position_last_season: int
    away_league_level_last_season: int

@dataclass
class BasicDatasetRow:
    match_id: int
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

@dataclass
class AggregatedDatasetRow:
    match_id: int
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