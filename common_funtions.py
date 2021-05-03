import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
from dataset_manager.class_definitions import AggregatedMatchData
from models import Match, Team, MatchResult


def get_scored_goals(matches: [Match], team: Team):
    return sum(match.full_time_home_goals for match in matches
               if match.home_team == team) + sum(match.full_time_away_goals for match in matches
                                                 if match.away_team == team)


def get_conceded_goals(matches: [Match], team: Team):
    return sum(match.full_time_away_goals for match in matches
               if match.home_team == team) + sum(match.full_time_home_goals for match in matches
                                                 if match.away_team == team)


def get_shots_fired(matches: [Match], team: Team):
    return sum(match.home_team_shots for match in matches
               if match.home_team == team) + sum(match.away_team_shots for match in matches
                                                 if match.away_team == team)


def get_shots_fired_on_target(matches: [Match], team: Team):
    return sum(match.home_team_shots_on_target for match in matches
               if match.home_team == team) + sum(match.away_team_shots_on_target for match in matches
                                                 if match.away_team == team)


def get_shots_conceded(matches: [Match], team: Team):
    return sum(match.away_team_shots for match in matches
               if match.home_team == team) + sum(match.home_team_shots for match in matches
                                                 if match.away_team == team)


def get_shots_conceded_on_target(matches: [Match], team: Team):
    return sum(match.away_team_shots_on_target for match in matches
               if match.home_team == team) + sum(match.home_team_shots_on_target for match in matches
                                                 if match.away_team == team)


def fill_last_matches_stats(matches: [Match], team: Team):
    matches_count = matches.count()
    if matches_count == 0:
        return AggregatedMatchData()
    return AggregatedMatchData(wins=(sum(1 for match in matches
                                         if (match.full_time_result == MatchResult.HOME_WIN
                                             and match.home_team == team)
                                         or (match.full_time_result == MatchResult.AWAY_WIN
                                             and match.away_team == team)) / matches_count),
                               draws=(sum(1 for match in matches if match.full_time_result ==
                                          MatchResult.DRAW) / matches_count),
                               loses=(sum(1 for match in matches
                                          if (match.full_time_result == MatchResult.AWAY_WIN
                                              and match.home_team == team)
                                          or (match.full_time_result == MatchResult.HOME_WIN
                                              and match.away_team == team)) / matches_count),
                               scored_goals=get_scored_goals(matches, team) / matches_count,
                               conceded_goals=get_conceded_goals(matches, team) / matches_count,
                               shots_fired=get_shots_fired(matches, team) / matches_count,
                               shots_fired_on_target=get_shots_fired_on_target(matches, team) / matches_count,
                               shots_conceded=get_shots_conceded(matches, team) / matches_count,
                               shots_conceded_on_target=get_shots_conceded_on_target(matches, team) / matches_count)


def get_y_ready_for_learning(dataset: pd.DataFrame):
    y = dataset['result'].to_numpy()
    one_hot_y = to_categorical(y, num_classes=3)
    odds = dataset[['home_odds', 'draw_odds', 'away_odds']].to_numpy()
    zero_vector = np.zeros((one_hot_y.shape[0], 1))
    return np.float32(np.concatenate((one_hot_y, zero_vector, odds), axis=1))

# def get_y_ready_for_learning(y: np.ndarray, odds: np.ndarray):
#     one_hot_y = to_categorical(y, num_classes=3)
#     zero_vector = np.zeros((one_hot_y.shape[0], 1))
#     return np.float32(np.concatenate((one_hot_y, zero_vector, odds), axis=1))
