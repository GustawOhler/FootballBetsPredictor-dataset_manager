from datetime import datetime
from os.path import isfile, getmtime
from random import sample
import keyboard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from constants import ids_path, curr_dataset_name, dataset_with_ext, dataset_path, is_model_rnn, TAKE_MATCHES_FROM_QUERY, SPLIT_MATCHES_BY_QUERY, DatasetType, \
    dataset_ext, curr_dataset, SHOULD_DROP_ODDS_FROM_DATASET
from dataset_manager.basic_dataset_creator import BasicDatasetCreator
from dataset_manager.dataset_with_aggregated_matches_creator import DatasetWithAggregatedMatchesCreator
from dataset_manager.dataset_with_separated_matches_creator import DatasetWithSeparatedMatchesCreator
from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.common_funtions import get_y_ready_for_learning, get_nn_input_attrs
import matplotlib.pyplot as plt
from models import Match, Season, League
import hashlib

wanted_match_ids_query = Match.select(Match.id).join(Season).join(League).where((Match.average_home_odds != 0.0) & (Match.average_away_odds != 0.0)
                                                                                & (Match.average_draw_odds != 0.0))
val_match_ids_query = Match.select(Match.id).join(Season).join(League).where((Match.date > datetime(2017, 6, 30)) & (Match.date <
                                                                                                                     datetime(2020, 1, 1))
                                                                             & (League.division == 1))


def show_dataset_histogram(dataset: pd.DataFrame):
    column_names = dataset.columns.values
    curr_column_index = 0
    print(column_names[curr_column_index], end="\r")

    while True:
        pressed_key = keyboard.read_key()
        if pressed_key == 'down':
            curr_column_index = curr_column_index + 1 if curr_column_index < column_names.shape[0] - 1 else 0
            print(column_names[curr_column_index], end="\r")
        elif pressed_key == 'up':
            curr_column_index = curr_column_index - 1 if curr_column_index > 0 else column_names.shape[0] - 1
            print(column_names[curr_column_index], end="\r")
        elif pressed_key == 'enter':
            dataset[column_names[curr_column_index]].hist(bins=50)
            plt.title(column_names[curr_column_index])
            plt.show()
        elif pressed_key == 'esc':
            break
        import time
        time.sleep(0.25)


def get_dataset_path(dataset_type: DatasetType):
    local_dataset_path = 'dataset_manager/datasets/' + dataset_type.value
    return local_dataset_path + dataset_ext


def load_dataset(dataset_path_with_ext=dataset_with_ext):
    return pd.read_csv(dataset_path_with_ext)


def get_query_hash_string(split_query_only: bool = False):
    hashes = ''
    if TAKE_MATCHES_FROM_QUERY and not split_query_only:
        sql_query_whole = wanted_match_ids_query.sql()[0]
        hashes += '_' + hashlib.sha1(sql_query_whole.encode("UTF-8")).hexdigest()[:8]
    if SPLIT_MATCHES_BY_QUERY:
        sql_query_split = val_match_ids_query.sql()[0]
        hashes += '_' + hashlib.sha1(sql_query_split.encode("UTF-8")).hexdigest()[:8]
    return hashes


def get_splitted_dataset_path(type: DatasetSplit):
    return dataset_path + '_' + type.value + '_split' + get_query_hash_string() + '.csv'


def get_splitted_ids_path(type: DatasetSplit):
    return ids_path + '_' + type.value + get_query_hash_string(True) + '.txt'


def save_splitted_dataset(x, y, type: DatasetSplit, column_names):
    concat = np.column_stack((x, y))
    df = pd.DataFrame(data=concat, columns=column_names)
    df.to_csv(get_splitted_dataset_path(type), index=False, float_format='%.3f')
    return df


def save_splitted_dataset(dataset: pd.DataFrame, type: DatasetSplit):
    dataset.to_csv(get_splitted_dataset_path(type), index=False, float_format='%.3f')


def load_splitted_dataset(type: DatasetSplit):
    dataset = pd.read_csv(get_splitted_dataset_path(type))
    x = get_nn_input_attrs(dataset, type, is_model_rnn)
    y = get_y_ready_for_learning(dataset)
    return x, y


def save_splitted_match_ids(train_ids, val_ids, test_ids):
    np.savetxt(get_splitted_ids_path(DatasetSplit.TRAIN), train_ids, fmt='%d')
    np.savetxt(get_splitted_ids_path(DatasetSplit.VAL), val_ids, fmt='%d')
    if test_ids is not None:
        np.savetxt(get_splitted_ids_path(DatasetSplit.TEST), test_ids, fmt='%d')


def load_splitted_match_ids(type: DatasetSplit):
    return np.loadtxt(get_splitted_ids_path(type))


def load_ids_in_right_order(type: DatasetSplit):
    dataset = pd.read_csv(get_splitted_dataset_path(type))
    return dataset['match_id'].to_numpy()


def save_new_splits(train_set, val_set, test_set):
    if test_set is not None:
        save_splitted_match_ids(train_set['match_id'].to_numpy(), val_set['match_id'].to_numpy(), test_set['match_id'].to_numpy())
        save_splitted_dataset(test_set, DatasetSplit.TEST)
    else:
        save_splitted_match_ids(train_set['match_id'].to_numpy(), val_set['match_id'].to_numpy(), None)
    save_splitted_dataset(train_set, DatasetSplit.TRAIN)
    save_splitted_dataset(val_set, DatasetSplit.VAL)


def split_dataset_by_query(dataset: pd.DataFrame, validation_split: float, test_split: float):
    wanted_quantity = int(validation_split * dataset.shape[0])
    match_ids = [match.id for match in val_match_ids_query]
    if len(match_ids) > 0:
        validation_candidate = dataset.loc[dataset['match_id'].isin(match_ids)]
        if validation_candidate.shape[0] > wanted_quantity * 1.25:
            indexes_to_take = sample(range(0, validation_candidate.shape[0]), wanted_quantity)
            validation_candidate = validation_candidate.iloc[indexes_to_take]
        train_set = dataset.loc[~dataset['match_id'].isin(validation_candidate['match_id'])]
        test_set = None
        if test_split > 0.0:
            indexes_to_take_for_test = sample(range(0, validation_candidate.shape[0]), int(validation_candidate.shape[0] * test_split))
            test_set = validation_candidate.iloc[indexes_to_take_for_test]
            validation_candidate = validation_candidate.drop(validation_candidate.index[indexes_to_take_for_test])
        save_new_splits(train_set, validation_candidate, test_set)
        returned_datasets = [(get_nn_input_attrs(train_set, DatasetSplit.TRAIN, is_model_rnn), get_y_ready_for_learning(train_set)), (
            get_nn_input_attrs(validation_candidate, DatasetSplit.VAL, is_model_rnn), get_y_ready_for_learning(validation_candidate))]
        if test_set is not None:
            returned_datasets.append((get_nn_input_attrs(test_set, DatasetSplit.TEST, is_model_rnn), get_y_ready_for_learning(test_set)))
        return returned_datasets
    else:
        return split_dataset(dataset, validation_split)


def split_dataset(dataset: pd.DataFrame, validation_split: float, test_split: float):
    test_set = None
    if test_split > 0.0:
        train_dataset, val_dataset, test_set = split_stratified_into_train_val_test(dataset, frac_train=1.0-validation_split-test_split,
                                                                                    frac_val=validation_split*(1.0-test_split),
                                                                                    frac_test=validation_split*test_split)
    else:
        train_dataset, val_dataset = train_test_split(dataset, test_size=validation_split, stratify=dataset[['result']])
    save_new_splits(train_dataset, val_dataset, test_set)
    returned_datasets = [(get_nn_input_attrs(train_dataset, DatasetSplit.TRAIN, is_model_rnn), get_y_ready_for_learning(train_dataset)), (
        get_nn_input_attrs(val_dataset, DatasetSplit.VAL, is_model_rnn), get_y_ready_for_learning(val_dataset))]
    if test_set is not None:
        returned_datasets.append((get_nn_input_attrs(test_set, DatasetSplit.TEST, is_model_rnn), get_y_ready_for_learning(test_set)))
    return returned_datasets


def split_stratified_into_train_val_test(df_input, stratify_colname='result',
                                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def split_dataset_from_ids(dataset: pd.DataFrame):
    train_dataset_path = get_splitted_dataset_path(DatasetSplit.TRAIN)
    val_dataset_path = get_splitted_dataset_path(DatasetSplit.VAL)
    test_dataset_path = get_splitted_dataset_path(DatasetSplit.TEST)
    train_ids_path = get_splitted_ids_path(DatasetSplit.TRAIN)
    if isfile(train_dataset_path) and isfile(val_dataset_path) \
            and getmtime(train_dataset_path) >= getmtime(train_ids_path):
        returned_datasets = [load_splitted_dataset(DatasetSplit.TRAIN), load_splitted_dataset(DatasetSplit.VAL)]
        if isfile(test_dataset_path):
            returned_datasets.append(load_splitted_dataset(DatasetSplit.TEST))
        return returned_datasets
    else:
        train_ids = load_splitted_match_ids(DatasetSplit.TRAIN)
        val_ids = load_splitted_match_ids(DatasetSplit.VAL)
        train_dataset = dataset.loc[dataset['match_id'].isin(train_ids)]
        val_dataset = dataset.loc[dataset['match_id'].isin(val_ids)]
        save_splitted_dataset(train_dataset, DatasetSplit.TRAIN)
        save_splitted_dataset(val_dataset, DatasetSplit.VAL)
        x_train = get_nn_input_attrs(train_dataset, DatasetSplit.TRAIN, is_model_rnn)
        # y_train = train_dataset['result']
        x_val = get_nn_input_attrs(val_dataset, DatasetSplit.VAL, is_model_rnn)
        # y_val = val_dataset['result']
        returned_dataset = [(x_train, get_y_ready_for_learning(train_dataset)), (x_val, get_y_ready_for_learning(val_dataset))]
        try:
            test_ids = load_splitted_match_ids(DatasetSplit.TEST)
            test_dataset = dataset.loc[dataset['match_id'].isin(test_ids)]
            save_splitted_dataset(test_dataset, DatasetSplit.TEST)
            x_test = get_nn_input_attrs(test_dataset, DatasetSplit.TEST, is_model_rnn)
            returned_dataset.append((x_test, get_y_ready_for_learning(test_dataset)))
        finally:
            return returned_dataset


def get_matches_with_id(dataset: pd.DataFrame):
    if TAKE_MATCHES_FROM_QUERY:
        match_ids = [match.id for match in wanted_match_ids_query]
        if len(match_ids) > 0:
            return dataset.loc[dataset['match_id'].isin(match_ids)]
    return dataset


def get_splitted_dataset(should_generate_dataset: bool, should_create_new_split: bool, validation_to_train_split_ratio: float,
                         test_to_validation_split: float):
    if should_generate_dataset:
        dataset_creator = (globals()[curr_dataset_name])()
        dataset_creator.gather_data()
        dataset_creator.save_dataset_to_csv()
        dataset = dataset_creator.pandas_dataset
        dataset = get_matches_with_id(dataset)
        if should_create_new_split:
            if SPLIT_MATCHES_BY_QUERY:
                return split_dataset_by_query(dataset, validation_to_train_split_ratio, test_to_validation_split)
            else:
                return split_dataset(dataset, validation_to_train_split_ratio, test_to_validation_split)
        else:
            return split_dataset_from_ids(dataset)
    else:
        if should_create_new_split:
            dataset = get_matches_with_id(load_dataset())
            if SPLIT_MATCHES_BY_QUERY:
                return split_dataset_by_query(dataset, validation_to_train_split_ratio, test_to_validation_split)
            else:
                return split_dataset(dataset, validation_to_train_split_ratio, test_to_validation_split)
        else:
            dataset = get_matches_with_id(load_dataset())
            return split_dataset_from_ids(dataset)


def get_whole_dataset(should_generate_dataset: bool, dataset_type: DatasetType = curr_dataset):
    if should_generate_dataset:
        dataset_creator = (globals()[curr_dataset_name])()
        dataset_creator.gather_data()
        dataset_creator.save_dataset_to_csv()
        dataset = dataset_creator.pandas_dataset
    else:
        dataset = load_dataset(get_dataset_path(dataset_type))
    return get_matches_with_id(dataset)
    # return get_nn_input_attrs(dataset, DatasetSplit.WHOLE, is_model_rnn), get_y_ready_for_learning(dataset)


def get_dataset_ready_to_learn(dataset, ds_split, is_for_rnn=is_model_rnn, should_drop_odds=SHOULD_DROP_ODDS_FROM_DATASET):
    return get_nn_input_attrs(dataset, ds_split, is_for_rnn, should_drop_odds), get_y_ready_for_learning(dataset)
