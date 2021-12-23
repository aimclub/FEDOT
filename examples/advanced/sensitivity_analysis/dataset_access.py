from os.path import join

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def get_scoring_data():
    file_path_train = join('cases', 'data', 'scoring', 'scoring_train.csv')
    full_path_train = join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = join('cases', 'data', 'scoring', 'scoring_test.csv')
    full_path_test = join(str(fedot_project_root()), file_path_test)
    task = Task(TaskTypesEnum.classification)
    train = InputData.from_csv(full_path_train, task=task)
    test = InputData.from_csv(full_path_test, task=task)

    return train, test


def get_kc2_data():
    file_path = join('cases', 'data', 'kc2', 'kc2.csv')
    full_path = join(str(fedot_project_root()), file_path)
    task = Task(TaskTypesEnum.classification)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


def get_cholesterol_data():
    file_path = join('cases', 'data', 'cholesterol', 'cholesterol.csv')
    full_path = join(str(fedot_project_root()), file_path)
    task = Task(TaskTypesEnum.regression)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test
