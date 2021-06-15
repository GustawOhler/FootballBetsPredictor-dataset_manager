from abc import ABC, abstractmethod
from constants import dataset_with_ext
from models import Match
from dataclasses import asdict
from timeit import default_timer as timer
import pandas as pd
from flatten_dict import flatten

class BaseDatasetCreator(ABC):
    def __init__(self):
        self.matches_to_process = Match.select()
        self.dataset_objects = []
        self.pandas_dataset = None

    @abstractmethod
    def gather_data(self):
        pass

    def save_dataset_to_csv(self):
        csv_proccesing_start = timer()
        self.pandas_dataset = pd.DataFrame(flatten(asdict(row), reducer='underscore') for row in self.dataset_objects)
        self.pandas_dataset.to_csv(self.dataset_with_ext(), index=False, float_format='%.3f')
        csv_proccesing_end = timer()
        print("Czas przetwarzania rekordow do csvki: " + str("{:.2f} s".format(csv_proccesing_end - csv_proccesing_start)))

    def create_csv_with_dataset(self):
        self.gather_data()
        self.save_dataset_to_csv()

    def dataset_with_ext(self):
        return dataset_with_ext
        # return 'test.csv'