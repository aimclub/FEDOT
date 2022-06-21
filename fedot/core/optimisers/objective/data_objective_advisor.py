import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection._split import _BaseKFold
from typing import Type

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TaskTypesEnum


class DataObjectiveAdvisor:
    def __init__(self, threshold: float = 0.5):
        """
        Advisor for DataObjectiveBuilder for choice some parameters based on input_data

        :param threshold: threshold level for difference between uniform probabilities and real probabilities
        """
        self.threshold = threshold

    def propose_kfold(self, input_data: InputData) -> Type[_BaseKFold]:
        """
        Method to choose he most suitable strategy for making folds

        :param input_data: data to analyse
        """
        if input_data.task.task_type is TaskTypesEnum.classification and self.check_imbalance(input_data):
            return StratifiedKFold
        else:
            return KFold

    def check_imbalance(self, input_data: InputData) -> bool:
        """
        Checks data for imbalance - if probability of any class lower than uniform probability in threshold times it
        returns true
        :param input_data: data to analyse

        """
        _, counts = np.unique(input_data.target, return_counts=True)
        probabilities = counts / input_data.target.shape[0]
        uniform_probability = 1 / input_data.num_classes
        return np.any(np.abs(uniform_probability - probabilities) / uniform_probability > self.threshold)
