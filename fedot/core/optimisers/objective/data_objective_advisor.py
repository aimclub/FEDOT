import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TaskTypesEnum


class DataObjectiveAdvisor:
    def __init__(self, threshold: int = 0.5):
        self.threshold = threshold

    def propose_kfold(self, input_data: InputData):
        if input_data.task.task_type is TaskTypesEnum.classification and self.check_imbalance(input_data):
            return StratifiedKFold
        else:
            return KFold

    def check_imbalance(self, input_data):
        _, counts = np.unique(input_data.target, return_counts=True)
        probabilities = counts / input_data.target.shape[0]
        uniform_probability = 1 / input_data.num_classes
        return np.all((uniform_probability - probabilities) / uniform_probability < self.threshold)
