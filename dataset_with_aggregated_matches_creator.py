from dataclasses import asdict
from timeit import default_timer as timer
import pandas as pd
from flatten_dict import flatten
from dataset_manager.base_dataset_creator import BaseDatasetCreator
from dataset_manager.class_definitions import NNDatasetRow, results_dict
from dataset_manager.common_funtions import fill_last_matches_stats
from constants import dataset_with_ext
from models import Match, Table, TableTeam


class DatasetWithAggregatedMatchesCreator(BaseDatasetCreator):
    def gather_data(self):
        root_matches_count = self.matches_to_process.count()
        sum_of_time_elapsed = 0
        for index, root_match in enumerate(self.matches_to_process.iterator()):
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
            home_last_5_matches_as_home = Match.select().where(
                (Match.date < root_match.date) &
                (Match.home_team == root_home_team)).order_by(Match.date.desc()).limit(5)
            away_last_5_matches = Match.select().where(
                (Match.date < root_match.date) &
                ((Match.home_team == root_away_team)
                 | (Match.away_team == root_away_team))).order_by(Match.date.desc()).limit(5)
            away_last_5_matches_as_away = Match.select().where(
                (Match.date < root_match.date) &
                (Match.away_team == root_away_team)).order_by(Match.date.desc()).limit(5)
            last_3_matches_between_teams = Match.select().where(
                (Match.date < root_match.date) &
                (((Match.home_team == root_home_team) & (Match.away_team == root_away_team)) |
                 ((Match.home_team == root_away_team) & (Match.away_team == root_home_team)))).order_by(Match.date.desc()).limit(3)
            if home_last_5_matches.count() < 2 or away_last_5_matches.count() < 2 or home_team_table_stats.matches_played < 2 or \
                    away_team_table_stats.matches_played < 2:
                continue
            dataset_row = NNDatasetRow(match_id=root_match.id,
                                       home_position=home_team_table_stats.position, home_played_matches=home_team_table_stats.matches_played,
                                       home_wins=home_team_table_stats.wins / home_team_table_stats.matches_played,
                                       home_draws=home_team_table_stats.draws / home_team_table_stats.matches_played,
                                       home_loses=home_team_table_stats.loses / home_team_table_stats.matches_played,
                                       home_goals_scored=home_team_table_stats.goals_scored / home_team_table_stats.matches_played,
                                       home_goals_conceded=home_team_table_stats.goals_conceded / home_team_table_stats.matches_played,
                                       home_goal_difference=home_team_table_stats.goal_difference,
                                       home_last_5_matches=fill_last_matches_stats(home_last_5_matches, root_home_team),
                                       home_last_5_matches_at_home=fill_last_matches_stats(home_last_5_matches_as_home, root_home_team),
                                       away_position=away_team_table_stats.position, away_played_matches=away_team_table_stats.matches_played,
                                       away_wins=away_team_table_stats.wins / away_team_table_stats.matches_played,
                                       away_draws=away_team_table_stats.draws / away_team_table_stats.matches_played,
                                       away_loses=away_team_table_stats.loses / away_team_table_stats.matches_played,
                                       away_goals_scored=away_team_table_stats.goals_scored / away_team_table_stats.matches_played,
                                       away_goals_conceded=away_team_table_stats.goals_conceded / away_team_table_stats.matches_played,
                                       away_goal_difference=away_team_table_stats.goal_difference,
                                       away_last_5_matches=fill_last_matches_stats(away_last_5_matches, root_away_team),
                                       away_last_5_matches_at_away=fill_last_matches_stats(away_last_5_matches_as_away, root_away_team),
                                       result=results_dict[root_match.full_time_result.value], home_odds=root_match.average_home_odds,
                                       draw_odds=root_match.average_draw_odds,
                                       away_odds=root_match.average_away_odds,
                                       last_3_matches_between_teams=fill_last_matches_stats(last_3_matches_between_teams, root_home_team))
            self.dataset_objects.append(dataset_row)
            sum_of_time_elapsed = sum_of_time_elapsed + timer() - row_create_start
            index_from_1 = index + 1
            print("Przetwarzany rekord " + str(index_from_1) + " z " + str(root_matches_count) + " czyli "
                  + str("{:.2f}".format(index_from_1 * 100 / root_matches_count)) + "%. Sredni czas przetwarzania dla 100 rekordow: " + str(
                "{:.2f} s".format(sum_of_time_elapsed * 100 / index_from_1)), end=("\r" if index_from_1 != root_matches_count else "\n"))

    def save_dataset_to_csv(self):
        csv_proccesing_start = timer()
        self.pandas_dataset = pd.DataFrame(flatten(asdict(row), reducer='underscore') for row in self.dataset_objects)
        self.pandas_dataset.to_csv(dataset_with_ext, index=False, float_format='%.3f')
        csv_proccesing_end = timer()
        print("Czas przetwarzania rekordow do csvki: " + str("{:.2f} s".format(csv_proccesing_end - csv_proccesing_start)))
