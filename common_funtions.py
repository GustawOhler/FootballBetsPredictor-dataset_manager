import re
from random import randrange
from typing import List
import numpy as np
import pandas as pd
from peewee import DateTimeField
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from dataset_manager.class_definitions import AggregatedMatchData, SingleMatchForRootData, results_dict, DatasetSplit
from constants import dataset_with_ext, SHOULD_DROP_ODDS_FROM_DATASET
from models import Match, Team, MatchResult, TableTeam, Table


def get_scored_goals(matches: [Match], team: Team, half_time: bool = False):
    if half_time:
        return sum(match.half_time_home_goals for match in matches
                   if match.home_team == team) + sum(match.half_time_away_goals for match in matches
                                                     if match.away_team == team)
    return sum(match.full_time_home_goals for match in matches
               if match.home_team == team) + sum(match.full_time_away_goals for match in matches
                                                 if match.away_team == team)


def get_conceded_goals(matches: [Match], team: Team, half_time: bool = False):
    if half_time:
        return sum(match.half_time_away_goals for match in matches
                   if match.home_team == team) + sum(match.half_time_home_goals for match in matches
                                                     if match.away_team == team)
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


def get_team_property(matches: [Match], team: Team, property: str):
    return sum(getattr(match, "home_" + property) for match in matches
               if match.home_team == team) + sum(getattr(match, "away_" + property) for match in matches
                                                 if match.away_team == team)


def get_opp_property(matches: [Match], team: Team, property: str):
    return sum(getattr(match, "away_" + property) for match in matches
               if match.home_team == team) + sum(getattr(match, "home_" + property) for match in matches
                                                 if match.away_team == team)


def get_team_cards_summed(matches: [Match], team: Team):
    return get_team_property(matches, team, 'team_yellow_cards') + 2 * get_team_property(matches, team, 'team_red_cards')


def get_opp_cards_summed(matches: [Match], team: Team):
    return get_opp_property(matches, team, 'team_yellow_cards') + 2 * get_opp_property(matches, team, 'team_red_cards')


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


def get_result_from_perspective(result: MatchResult, is_my_team_home: bool):
    if is_my_team_home or result == MatchResult.DRAW:
        return results_dict[result.value]
    elif result == MatchResult.HOME_WIN:
        return results_dict[MatchResult.AWAY_WIN.value]
    elif result == MatchResult.AWAY_WIN:
        return results_dict[MatchResult.HOME_WIN.value]


def create_match_infos(matches: List[Match], team: Team, date_of_root_match: DateTimeField, how_many_expected: int):
    result_list = []
    for curr_match in matches:
        is_home = curr_match.home_team == team
        table_before_match = Table.get((Table.season == curr_match.season) & (Table.date == curr_match.date.date()))
        root_team_table_stats = TableTeam.get(
            (TableTeam.team == team) & (TableTeam.table == table_before_match))
        opp_team = curr_match.away_team if is_home else curr_match.home_team
        opp_team_table_stats = TableTeam.get(
            (TableTeam.team == opp_team) & (TableTeam.table == table_before_match))
        result_list.append(SingleMatchForRootData(
            root_result=get_result_from_perspective(curr_match.full_time_result, is_home), is_home=int(is_home),
            days_since_match=(date_of_root_match - curr_match.date).days,
            league_level=curr_match.season.league.division, position=root_team_table_stats.position, opposite_team_position=opp_team_table_stats.position,
            scored_goals=get_scored_goals([curr_match], team), conceded_goals=get_conceded_goals([curr_match], team),
            shots_fired=get_shots_fired([curr_match], team), shots_fired_on_target=get_shots_fired_on_target([curr_match], team),
            shots_conceded=get_shots_conceded([curr_match], team), shots_conceded_on_target=get_shots_conceded_on_target([curr_match], team),
            corners_taken=get_team_property([curr_match], team, 'team_corners'), opposite_corners=get_opp_property([curr_match], team, 'team_corners'),
            fouls_commited=get_team_property([curr_match], team, 'team_fouls_committed'),
            opposite_fouls=get_opp_property([curr_match], team, 'team_fouls_committed'),
            cards=get_team_cards_summed([curr_match], team), opposite_cards=get_opp_cards_summed([curr_match], team),
            half_time_result=get_result_from_perspective(curr_match.half_time_result, is_home),
            half_time_scored_goals=get_scored_goals([curr_match], team, True),
            half_time_conceded_goals=get_conceded_goals([curr_match], team, True)
        ))
    if len(result_list) < how_many_expected:
        for i in range(how_many_expected - len(result_list)):
            result_list.append(SingleMatchForRootData())
    return result_list


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

home_scalers = []
away_scalers = []
rest_scaler = RobustScaler()
one_big_scaler = RobustScaler()

def get_nn_input_attrs(dataset: pd.DataFrame, dataset_type: DatasetSplit, is_for_rnn):
    dropped_basic_fields_dataset = dataset.drop('result', axis='columns').drop('match_id', axis='columns')
    if SHOULD_DROP_ODDS_FROM_DATASET:
        dropped_basic_fields_dataset = dropped_basic_fields_dataset.drop('home_odds', axis='columns').drop('draw_odds', axis='columns') \
            .drop('away_odds', axis='columns')
    if is_for_rnn:
        dataset_without_half_results = dataset[dataset.columns.drop(dataset.filter(regex='.*half.*'))]
        home_data = np.stack(list(dataset_without_half_results[list(filter(lambda x: re.match('home_last_4_matches_' + str(i) + '.*', x),
                                                                           dataset_without_half_results.columns.values))]
                                  .to_numpy(dtype='float32')
                                  for i in reversed(range(4))), axis=1)
        away_data = np.stack(list(dataset_without_half_results[list(filter(lambda x: re.match('away_last_4_matches_' + str(i) + '.*', x),
                                                                           dataset_without_half_results.columns.values))]
                                  .to_numpy(dtype='float32')
                                  for i in reversed(range(4))), axis=1)
        # home_data = np.stack(list(dataset[list(filter(lambda x: re.match('home_last_4_matches_' + str(i) + '.*', x), dataset.columns.values))]
        #                           .to_numpy(dtype='float32')
        #                           for i in reversed(range(4))), axis=1)
        # away_data = np.stack(list(dataset[list(filter(lambda x: re.match('away_last_4_matches_' + str(i) + '.*', x), dataset.columns.values))]
        #                           .to_numpy(dtype='float32')
        #                           for i in reversed(range(4))), axis=1)
        rest_of_data = dropped_basic_fields_dataset.drop(list(filter(lambda x: re.match(r'(home|away)_last_4_matches_\d.*', x), dataset.columns.values))
                                                         , axis='columns').to_numpy(dtype='float32')

        all_data = np.concatenate([home_data, away_data], axis=0)
        all_data_reshaped = np.reshape(all_data, (all_data.shape[0] * all_data.shape[1], all_data.shape[2]))
        if dataset_type in [DatasetSplit.TRAIN, DatasetSplit.WHOLE]:
            one_big_scaler.fit(all_data_reshaped)

        for i in range(home_data.shape[1]):
            if dataset_type in [DatasetSplit.TRAIN, DatasetSplit.WHOLE]:
                home_data[:, i, :] = one_big_scaler.transform(home_data[:, i, :])
                away_data[:, i, :] = one_big_scaler.transform(away_data[:, i, :])
                # home_scalers.append(RobustScaler())
                # away_scalers.append(RobustScaler())
                # home_data[:, i, :] = home_scalers[i].fit_transform(home_data[:, i, :])
                # away_data[:, i, :] = away_scalers[i].fit_transform(away_data[:, i, :])
            else:
                # home_data[:, i, :] = home_scalers[i].transform(home_data[:, i, :])
                # away_data[:, i, :] = away_scalers[i].transform(away_data[:, i, :])
                home_data[:, i, :] = one_big_scaler.transform(home_data[:, i, :])
                away_data[:, i, :] = one_big_scaler.transform(away_data[:, i, :])
        # if dataset_type in [DatasetSplit.TRAIN, DatasetSplit.WHOLE]:
        #     rest_of_data = rest_scaler.fit_transform(rest_of_data)
        # else:
        #     rest_of_data = rest_scaler.transform(rest_of_data)
        return [home_data, away_data, rest_of_data]
    return dropped_basic_fields_dataset.to_numpy(dtype='float32')


def get_curr_dataset_column_names():
    column_names = pd.read_csv(dataset_with_ext, nrows=1).columns.tolist()
    column_names.remove('result')
    column_names.remove('match_id')
    if SHOULD_DROP_ODDS_FROM_DATASET:
        column_names.remove('home_odds')
        column_names.remove('draw_odds')
        column_names.remove('away_odds')
    return column_names


def get_random_row_with_db_object():
    dataset = pd.read_csv(dataset_with_ext)
    dataset_len = dataset.shape[0]
    return get_row_with_db_object(dataset, randrange(0, dataset_len))


def get_row_with_db_object(dataset, index):
    row = dataset.iloc[index]
    return {'dataset_object': row, 'db_object': Match.get_by_id(row['match_id'])}
