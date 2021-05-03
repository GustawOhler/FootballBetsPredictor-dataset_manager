from dataclasses import asdict
import pandas as pd
from flatten_dict import flatten
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from dataset_manager.class_definitions import DatasetType, NNDatasetRow
from dataset_manager.common_funtions import get_scored_goals, get_conceded_goals, get_y_ready_for_learning
from models import Match, Table, TableTeam, MatchResult

results_dict = {'H': 0, 'D': 1, 'A': 2}
dataset_path = 'dataset_manager/datasets/dataset'
dataset_with_ext = dataset_path + '.csv'


def create_dataset():
    dataset = []
    root_matches = Match.select()
    root_matches_count = root_matches.count()
    sum_of_time_elapsed = 0
    for index, root_match in enumerate(root_matches.iterator()):
        row_create_start = timer()
        root_home_team = root_match.home_team
        root_away_team = root_match.away_team
        table_before_match = Table.get(
            (Table.season == root_match.season) & (Table.date == root_match.date.date()))
        home_team_table_stats = TableTeam.get(
            (TableTeam.team == root_home_team) & (TableTeam.table == table_before_match))
        away_team_table_stats = TableTeam.get(
            (TableTeam.team == root_away_team) & (TableTeam.table == table_before_match))
        home_last_5_matches = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_home_team)
             | (Match.away_team == root_home_team))).order_by(Match.date.desc()).limit(5)
        away_last_5_matches = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_away_team)
             | (Match.away_team == root_away_team))).order_by(Match.date.desc()).limit(5)
        if home_last_5_matches.count() != 5 or away_last_5_matches.count() != 5 or home_team_table_stats.matches_played < 3 or away_team_table_stats.matches_played < 3:
            continue
        dataset_row = NNDatasetRow(home_position=home_team_table_stats.position, home_played_matches=home_team_table_stats.matches_played,
                                   home_wins=home_team_table_stats.wins,
                                   home_draws=home_team_table_stats.draws, home_loses=home_team_table_stats.loses,
                                   home_goals_scored=home_team_table_stats.goals_scored,
                                   home_goals_conceded=home_team_table_stats.goals_conceded, home_goal_difference=home_team_table_stats.goal_difference,
                                   home_team_wins_in_last_5_matches=sum(1 for match in home_last_5_matches
                                                                        if (match.full_time_result == MatchResult.HOME_WIN
                                                                            and match.home_team == root_home_team)
                                                                        or (match.full_time_result == MatchResult.AWAY_WIN
                                                                            and match.away_team == root_home_team)),
                                   home_team_draws_in_last_5_matches=sum(1 for match in home_last_5_matches if match.full_time_result == MatchResult.DRAW),
                                   home_team_loses_in_last_5_matches=sum(1 for match in home_last_5_matches
                                                                         if (match.full_time_result == MatchResult.AWAY_WIN
                                                                             and match.home_team == root_home_team)
                                                                         or (match.full_time_result == MatchResult.HOME_WIN
                                                                             and match.away_team == root_home_team)),
                                   home_team_scored_goals_in_last_5_matches=get_scored_goals(home_last_5_matches, root_home_team),
                                   home_team_conceded_goals_in_last_5_matches=get_conceded_goals(home_last_5_matches, root_home_team),
                                   away_position=away_team_table_stats.position, away_played_matches=away_team_table_stats.matches_played,
                                   away_wins=away_team_table_stats.wins,
                                   away_draws=away_team_table_stats.draws, away_loses=away_team_table_stats.loses,
                                   away_goals_scored=away_team_table_stats.goals_scored,
                                   away_goals_conceded=away_team_table_stats.goals_conceded, away_goal_difference=away_team_table_stats.goal_difference,
                                   away_team_wins_in_last_5_matches=sum(1 for match in away_last_5_matches
                                                                        if (match.full_time_result == MatchResult.HOME_WIN
                                                                            and match.home_team == root_away_team)
                                                                        or (match.full_time_result == MatchResult.AWAY_WIN
                                                                            and match.away_team == root_away_team)),
                                   away_team_draws_in_last_5_matches=sum(1 for match in away_last_5_matches
                                                                         if match.full_time_result == MatchResult.DRAW),
                                   away_team_loses_in_last_5_matches=sum(1 for match in away_last_5_matches
                                                                         if (match.full_time_result == MatchResult.AWAY_WIN
                                                                             and match.home_team == root_away_team)
                                                                         or (match.full_time_result == MatchResult.HOME_WIN
                                                                             and match.away_team == root_away_team)),
                                   away_team_scored_goals_in_last_5_matches=get_scored_goals(away_last_5_matches, root_away_team),
                                   away_team_conceded_goals_in_last_5_matches=get_conceded_goals(away_last_5_matches, root_away_team),
                                   result=results_dict[root_match.full_time_result.value], home_odds=root_match.average_home_odds,
                                   draw_odds=root_match.average_draw_odds,
                                   away_odds=root_match.average_away_odds)
        dataset.append(dataset_row)
        sum_of_time_elapsed = sum_of_time_elapsed + timer() - row_create_start
        index_from_1 = index + 1
        print("Przetwarzany rekord " + str(index_from_1) + " z " + str(root_matches_count) + " czyli "
              + str("{:.2f}".format(index_from_1 * 100 / root_matches_count)) + "%. Sredni czas przetwarzania dla 100 rekordow: " + str(
            "{:.2f} s".format(sum_of_time_elapsed * 100/index_from_1)), end=("\r" if index_from_1 != root_matches_count else "\n"))

    csv_proccesing_start = timer()
    pd_dataset = pd.DataFrame(flatten(asdict(row), reducer='underscore') for row in dataset)
    pd_dataset.to_csv(dataset_with_ext, index=False, float_format='%.3f')
    csv_proccesing_end = timer()
    print("Czas przetwarzania rekordow do csvki: " + str("{:.2f} s".format(csv_proccesing_end - csv_proccesing_start)))
    return pd_dataset


def load_dataset():
    return pd.read_csv(dataset_with_ext)


def save_splitted_dataset(x, y, type: DatasetType, column_names):
    concat = np.column_stack((x, y))
    df = pd.DataFrame(data=concat, columns=column_names)
    df.to_csv(dataset_path + '_' + type.value + '_split.csv', index=False, float_format='%.3f')
    return df


def load_splitted_dataset(type: DatasetType):
    dataset = pd.read_csv(dataset_path + '_' + type.value + '_split.csv')
    x = dataset.drop('result', axis='columns').to_numpy(dtype='float32')
    y = get_y_ready_for_learning(dataset)
    return x, y


def split_dataset(dataset: pd.DataFrame, validation_split):
    x = dataset.drop('result', axis='columns').to_numpy(dtype='float32')
    y = dataset['result'].to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)
    column_names = dataset.columns.values.tolist()
    column_names.remove('result')
    column_names.append('result')
    train_df = save_splitted_dataset(x_train, y_train, DatasetType.TRAIN, column_names)
    val_df = save_splitted_dataset(x_val, y_val, DatasetType.VAL, column_names)
    return (x_train, get_y_ready_for_learning(train_df)), (x_val, get_y_ready_for_learning(val_df))


def get_splitted_dataset(should_generate_dataset: bool, should_create_new_split: bool, validation_to_train_split_ratio: float):
    if should_generate_dataset:
        dataset = create_dataset()
        return split_dataset(dataset, validation_to_train_split_ratio)
    else:
        dataset = load_dataset()
        if should_create_new_split:
            return split_dataset(dataset, validation_to_train_split_ratio)
        else:
            x_train, y_train = load_splitted_dataset(DatasetType.TRAIN)
            x_val, y_val = load_splitted_dataset(DatasetType.VAL)
            return (x_train, y_train), (x_val, y_val)