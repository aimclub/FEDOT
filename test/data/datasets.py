import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cases.metocean_forecasting_problem import prepare_input_data
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from test.integration.models.test_split_train_test import get_synthetic_input_data
from test.unit.tasks.test_classification import get_iris_data, get_synthetic_classification_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.tasks.test_regression import get_synthetic_regression_data


def get_split_data_paths():
    file_path_train = 'test/data/simple_regression_train.csv'
    file_path_test = 'test/data/simple_regression_test.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    return full_path_train, full_path_test


def get_split_data():
    task_type = 'regression'
    train_full, test = get_split_data_paths()
    train_file = pd.read_csv(train_full)
    x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)
    return task_type, x_train, x_test, y_train, y_test


def get_cholesterol_dataset():
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'
    data = InputData.from_csv(data_path, task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)
    return train, test


def get_dataset(task_type: str, validation_blocks: Optional[int] = None, n_samples: int = 200,
                n_features: int = 8, forecast_length: int = 5, iris_dataset=True):
    if task_type == 'regression':
        data = get_synthetic_regression_data(n_samples=n_samples, n_features=n_features, random_state=42)
        train_data, test_data = train_test_data_setup(data)
        threshold = np.std(test_data.target) * 0.05
    elif task_type == 'classification':
        if iris_dataset:
            data = get_iris_data()
        else:
            data = get_synthetic_classification_data(n_samples=n_samples, n_features=n_features, random_state=42)
        train_data, test_data = train_test_data_setup(data, shuffle=True)
        threshold = 0.95
    elif task_type == 'clustering':
        data = get_synthetic_input_data(n_samples=100)
        train_data, test_data = train_test_data_setup(data)
        threshold = 0.5
    elif task_type == 'ts_forecasting':
        train_data, test_data = get_ts_data(forecast_length=forecast_length, validation_blocks=validation_blocks)
        threshold = np.std(test_data.target)
    else:
        raise ValueError('Incorrect type of machine learning task')
    return train_data, test_data, threshold


def get_multimodal_ts_data(size=500):
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    target_history, add_history, _ = prepare_input_data(full_path_train, full_path_test,
                                                        history_size=size)
    historical_data = {
        'ws': add_history,  # additional variable
        'ssh': target_history,  # target variable
    }
    return historical_data, target_history


def load_categorical_unimodal():
    dataset_path = 'test/data/classification_with_categorical.csv'
    full_path = os.path.join(str(fedot_project_root()), dataset_path)
    data = InputData.from_csv(full_path)
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    return train_data, test_data


def load_categorical_multidata():
    # Create features table
    features_first = np.array([[0, '  a'], [1, ' a '], [2, '  b'], [3, np.nan], [4, '  a'],
                               [5, '  b'], [6, 'b  '], [7, '  c'], [8, ' c ']], dtype=object)
    features_second = np.array([[10, '  a'], [11, ' a '], [12, '  b'], [13, ' a '], [14, '  a'],
                                [15, '  b'], [16, 'b  '], [17, '  c'], [18, ' c ']], dtype=object)
    # TODO @andreygetmanov (fails if target = ['true', 'false', ...])
    target = np.array([1, 0, 1, 0, 0, 0, 0, 1, 1])

    fit_data = {'first': features_first,
                'second': features_second}

    return fit_data, target


def data_with_binary_features_and_categorical_target():
    """
    A dataset is generated where features and target require transformations.
    The categorical binary features and categorical target must be converted to int
    """
    task = Task(TaskTypesEnum.classification)
    features = np.array([['red', 'blue'],
                         ['red', 'blue'],
                         ['red', 'blue'],
                         [np.nan, 'blue'],
                         ['green', 'blue'],
                         ['green', 'orange'],
                         ['red', 'orange']])
    target = np.array(['red-blue', 'red-blue', 'red-blue', 'red-blue', 'green-blue', 'green-orange', 'red-orange'])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5, 6], features=features, target=target,
                            task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData())

    return train_input
