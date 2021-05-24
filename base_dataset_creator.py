from abc import ABC, abstractmethod
from models import Match


class BaseDatasetCreator(ABC):
    def __init__(self):
        self.matches_to_process = Match.select()
        self.dataset_objects = []

    @abstractmethod
    def gather_data(self):
        pass

    @abstractmethod
    def save_dataset_to_csv(self):
        pass

    def create_csv_with_dataset(self):
        self.gather_data()
        self.save_dataset_to_csv()
