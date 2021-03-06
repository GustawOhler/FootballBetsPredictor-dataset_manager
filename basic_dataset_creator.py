from dataclasses import asdict
from timeit import default_timer as timer
import pandas as pd
from flatten_dict import flatten
from dataset_manager.base_dataset_creator import BaseDatasetCreator
from dataset_manager.class_definitions import DatasetWithSeparatedMatchesRow, results_dict, BasicDatasetRow
from dataset_manager.common_funtions import fill_last_matches_stats, get_scored_goals, get_conceded_goals
from models import Match, Table, TableTeam, MatchResult


class BasicDatasetCreator(BaseDatasetCreator):
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
            away_last_5_matches = Match.select().where(
                (Match.date < root_match.date) &
                ((Match.home_team == root_away_team)
                 | (Match.away_team == root_away_team))).order_by(Match.date.desc()).limit(5)
            if home_last_5_matches.count() != 5 or away_last_5_matches.count() != 5 or home_team_table_stats.matches_played < 3 or away_team_table_stats.matches_played < 3:
                continue
            dataset_row = BasicDatasetRow(match_id=root_match.id,
                                          home_position=home_team_table_stats.position, home_played_matches=home_team_table_stats.matches_played,
                                          home_wins=home_team_table_stats.wins,
                                          home_draws=home_team_table_stats.draws, home_loses=home_team_table_stats.loses,
                                          home_goals_scored=home_team_table_stats.goals_scored,
                                          home_goals_conceded=home_team_table_stats.goals_conceded, home_goal_difference=home_team_table_stats.goal_difference,
                                          home_team_wins_in_last_5_matches=sum(1 for match in home_last_5_matches
                                                                               if (match.full_time_result == MatchResult.HOME_WIN
                                                                                   and match.home_team == root_home_team)
                                                                               or (match.full_time_result == MatchResult.AWAY_WIN
                                                                                   and match.away_team == root_home_team)),
                                          home_team_draws_in_last_5_matches=sum(
                                              1 for match in home_last_5_matches if match.full_time_result == MatchResult.DRAW),
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
            self.dataset_objects.append(dataset_row)
            sum_of_time_elapsed = sum_of_time_elapsed + timer() - row_create_start
            index_from_1 = index + 1
            print("Przetwarzany rekord " + str(index_from_1) + " z " + str(root_matches_count) + " czyli "
                  + str("{:.2f}".format(index_from_1 * 100 / root_matches_count)) + "%. Sredni czas przetwarzania dla 100 rekordow: " + str(
                "{:.2f} s".format(sum_of_time_elapsed * 100 / index_from_1)), end=("\r" if index_from_1 != root_matches_count else "\n"))

