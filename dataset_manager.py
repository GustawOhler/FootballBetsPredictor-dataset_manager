from os.path import isfile, getmtime
import keyboard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from constants import ids_path, curr_dataset_name, dataset_with_ext, dataset_path, is_model_rnn
from dataset_manager.basic_dataset_creator import BasicDatasetCreator
from dataset_manager.dataset_with_aggregated_matches_creator import DatasetWithAggregatedMatchesCreator
from dataset_manager.dataset_with_separated_matches_creator import DatasetWithSeparatedMatchesCreator
from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.common_funtions import get_y_ready_for_learning, get_nn_input_attrs
import matplotlib.pyplot as plt

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

def load_dataset():
    return pd.read_csv(dataset_with_ext)


def get_splitted_dataset_path(type: DatasetSplit):
    return dataset_path + '_' + type.value + '_split.csv'


def get_splitted_ids_path(type: DatasetSplit):
    return ids_path + '_' + type.value + '.txt'


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


def save_splitted_match_ids(train_ids, val_ids):
    np.savetxt(get_splitted_ids_path(DatasetSplit.TRAIN), train_ids, fmt='%d')
    np.savetxt(get_splitted_ids_path(DatasetSplit.VAL), val_ids, fmt='%d')


def load_splitted_match_ids(type: DatasetSplit):
    return np.loadtxt(get_splitted_ids_path(type))


def load_ids_in_right_order(type: DatasetSplit):
    dataset = pd.read_csv(get_splitted_dataset_path(type))
    return dataset['match_id'].to_numpy()


def split_dataset(dataset: pd.DataFrame, validation_split):
    train_dataset, val_dataset = train_test_split(dataset, test_size=validation_split)
    save_splitted_dataset(train_dataset, DatasetSplit.TRAIN)
    save_splitted_dataset(val_dataset, DatasetSplit.VAL)
    save_splitted_match_ids(train_dataset['match_id'].to_numpy(), val_dataset['match_id'].to_numpy())
    return (get_nn_input_attrs(train_dataset, DatasetSplit.TRAIN, is_model_rnn), get_y_ready_for_learning(train_dataset)), (
        get_nn_input_attrs(val_dataset, DatasetSplit.VAL, is_model_rnn), get_y_ready_for_learning(val_dataset))


def split_dataset_from_ids(dataset: pd.DataFrame):
    train_dataset_path = dataset_path + '_' + DatasetSplit.TRAIN.value + '_split.csv'
    val_dataset_path = dataset_path + '_' + DatasetSplit.VAL.value + '_split.csv'
    train_ids_path = ids_path + '_' + DatasetSplit.TRAIN.value + '.txt'
    if isfile(train_dataset_path) and isfile(val_dataset_path)\
            and getmtime(train_dataset_path) > getmtime(train_ids_path):
        return load_splitted_dataset(DatasetSplit.TRAIN), load_splitted_dataset(DatasetSplit.VAL)
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
