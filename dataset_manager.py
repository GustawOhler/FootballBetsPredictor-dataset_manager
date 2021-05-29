from os.path import isfile, getmtime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from constants import ids_path, curr_dataset_name, dataset_with_ext, dataset_path
from dataset_manager.basic_dataset_creator import BasicDatasetCreator
from dataset_manager.dataset_with_aggregated_matches_creator import DatasetWithAggregatedMatchesCreator
from dataset_manager.dataset_with_separated_matches_creator import DatasetWithSeparatedMatchesCreator
from dataset_manager.class_definitions import DatasetType
from dataset_manager.common_funtions import get_y_ready_for_learning, get_nn_input_attrs


def load_dataset():
    return pd.read_csv(dataset_with_ext)


def get_splitted_dataset_path(type: DatasetType):
    return dataset_path + '_' + type.value + '_split.csv'


def get_splitted_ids_path(type: DatasetType):
    return ids_path + '_' + type.value + '.txt'


def save_splitted_dataset(x, y, type: DatasetType, column_names):
    concat = np.column_stack((x, y))
    df = pd.DataFrame(data=concat, columns=column_names)
    df.to_csv(get_splitted_dataset_path(type), index=False, float_format='%.3f')
    return df


def save_splitted_dataset(dataset: pd.DataFrame, type: DatasetType):
    dataset.to_csv(get_splitted_dataset_path(type), index=False, float_format='%.3f')


def load_splitted_dataset(type: DatasetType):
    dataset = pd.read_csv(get_splitted_dataset_path(type))
    x = get_nn_input_attrs(dataset)
    y = get_y_ready_for_learning(dataset)
    return x, y


def save_splitted_match_ids(train_ids, val_ids):
    np.savetxt(get_splitted_ids_path(DatasetType.TRAIN), train_ids, fmt='%d')
    np.savetxt(get_splitted_ids_path(DatasetType.VAL), val_ids, fmt='%d')


def load_splitted_match_ids(type: DatasetType):
    return np.loadtxt(get_splitted_ids_path(type))


def load_ids_in_right_order(type: DatasetType):
    dataset = pd.read_csv(get_splitted_dataset_path(type))
    return dataset['match_id'].to_numpy()


def split_dataset(dataset: pd.DataFrame, validation_split):
    train_dataset, val_dataset = train_test_split(dataset, test_size=validation_split)
    save_splitted_dataset(train_dataset, DatasetType.TRAIN)
    save_splitted_dataset(val_dataset, DatasetType.VAL)
    save_splitted_match_ids(train_dataset['match_id'].to_numpy(), val_dataset['match_id'].to_numpy())
    return (get_nn_input_attrs(train_dataset), get_y_ready_for_learning(train_dataset)), (
        get_nn_input_attrs(val_dataset), get_y_ready_for_learning(val_dataset))


def split_dataset_from_ids(dataset: pd.DataFrame):
    train_dataset_path = dataset_path + '_' + DatasetType.TRAIN.value + '_split.csv'
    val_dataset_path = dataset_path + '_' + DatasetType.VAL.value + '_split.csv'
    train_ids_path = ids_path + '_' + DatasetType.TRAIN.value + '.txt'
    if isfile(train_dataset_path) and isfile(val_dataset_path)\
            and getmtime(train_dataset_path) > getmtime(train_ids_path):
        return load_splitted_dataset(DatasetType.TRAIN), load_splitted_dataset(DatasetType.VAL)
    else:
        train_ids = load_splitted_match_ids(DatasetType.TRAIN)
        val_ids = load_splitted_match_ids(DatasetType.VAL)
        train_dataset = dataset.loc[dataset['match_id'].isin(train_ids)]
        val_dataset = dataset.loc[dataset['match_id'].isin(val_ids)]
        save_splitted_dataset(train_dataset, DatasetType.TRAIN)
        save_splitted_dataset(val_dataset, DatasetType.VAL)
        x_train = get_nn_input_attrs(train_dataset)
        # y_train = train_dataset['result']
        x_val = get_nn_input_attrs(val_dataset)
        # y_val = val_dataset['result']
        return (x_train, get_y_ready_for_learning(train_dataset)), (x_val, get_y_ready_for_learning(val_dataset))


def get_splitted_dataset(should_generate_dataset: bool, should_create_new_split: bool, validation_to_train_split_ratio: float):
    if should_generate_dataset:
        dataset_creator = (globals()[curr_dataset_name])()
        dataset_creator.gather_data()
        dataset_creator.save_dataset_to_csv()
        dataset = dataset_creator.pandas_dataset
        if should_create_new_split:
            return split_dataset(dataset, validation_to_train_split_ratio)
        else:
            return split_dataset_from_ids(dataset)
    else:
        if should_create_new_split:
            dataset = load_dataset()
            return split_dataset(dataset, validation_to_train_split_ratio)
        else:
            dataset = load_dataset()
            return split_dataset_from_ids(dataset)
