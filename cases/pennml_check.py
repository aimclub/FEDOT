import os
import pandas as pd
import numpy as np
from typing import List

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


# This script for check if everything is ok with launches on PennML datasets
# The datasets must be placed in cases/data/pennml folder
def data_setup(predictors, target, task: TaskTypesEnum = TaskTypesEnum.classification,
               data_type=DataTypesEnum.table):
    """ Train test splitting function """
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(target)
    data = InputData(features=predictors, target=target, idx=np.arange(0, len(target)),
                     task=Task(task),
                     data_type=data_type)
    train_data, test_data = train_test_data_setup(data, split_ratio=0.8)
    return train_data, test_data


def run_classification_exp(dataset_numbers: List[int]):
    """ Start classification task with desired dataset

    :param dataset_numbers: list with ids of the dataset in the folder
    """
    data_path = os.path.join(fedot_project_root(), 'cases', 'data', 'pennml')
    datasets = os.listdir(data_path)

    for dataset_number in dataset_numbers:
        dataset_name = datasets[dataset_number]
        print(f'Processing dataset with name {dataset_name}')

        dataset_path = os.path.join(data_path, dataset_name)
        data = pd.read_csv(dataset_path)
        data = data.head(500)

        predictors = np.array(data.iloc[:, :-1])
        target = np.array(data.iloc[:, -1])

        train_data, test_data = data_setup(predictors, target)

        fedot = Fedot(problem='classification', timeout=0.5)
        fedot.fit(features=train_data)
        print(fedot.predict(test_data))


if __name__ == '__main__':
    run_classification_exp(dataset_numbers=[0])
