import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.api.api_runner import Fedot, check_data_type
from fedot.core.data.data import train_test_data_setup, InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root
from test.unit.models.test_split_train_test import get_synthetic_input_data
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_synthetic_ts_data_linear
from test.unit.tasks.test_regression import get_synthetic_regression_data

composer_params = {'max_depth': 1,
                   'max_arity': 2,
                   'learning_time': 0.0001}


def get_split_data_paths():
    file_path_train = 'test/data/simple_regression_train.csv'
    file_path_test = 'test/data/simple_regression_test.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    return full_path_train, full_path_test


def get_split_data():
    task_type = 'regression'
    train_full, test = get_split_data_paths()
    train_file = pd.read_csv(train_full)
    x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)
    return task_type, x_train, x_test, y_train, y_test


def get_dataset(task_type: str):
    if task_type == 'regression':
        data = get_synthetic_regression_data()
        train_data, test_data = train_test_data_setup(data)
        threshold = np.std(test_data.target) * 0.05
    elif task_type == 'classification':
        data = get_iris_data()
        train_data, test_data = train_test_data_setup(data, shuffle_flag=True)
        threshold = 0.95
    elif task_type == 'clustering':
        data = get_synthetic_input_data(n_samples=10000)
        train_data, test_data = train_test_data_setup(data)
        threshold = 0.5
    elif task_type == 'ts_forecasting':
        train_data, test_data = get_synthetic_ts_data_linear(forecast_length=1, max_window_size=10)
        threshold = np.str(test_data.target)
    else:
        raise ValueError('Incorrect type of machine learning task')
    return train_data, test_data, threshold


def test_api_predict_correct(task_type: str = 'classification'):
    train_data, test_data, threshold = get_dataset(task_type)
    model = Fedot(ml_task=task_type,
                  composer_params=composer_params)
    fedot_model = model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.quality_metric()
    assert type(fedot_model) != Fedot
    assert type(prediction) != np.array
    assert type(metric) != float
    assert metric < threshold


def test_api_forecast_correct(task_type: str = 'ts_forecasting'):
    train_data, test_data, threshold = get_dataset(task_type)
    model = Fedot(ml_task='ts_forecasting')
    fedot_model = model.fit(features=train_data)
    ts_forecast = model.forecast(pre_history=test_data, forecast_length=1)
    metric = model.quality_metric()
    assert type(fedot_model) != Fedot
    assert type(ts_forecast) != np.array
    assert type(metric) != float
    assert metric < threshold


def test_api_check_data_correct():
    task_type, x_train, x_test, y_train, y_test = get_split_data()
    path_to_train, path_to_test = get_split_data_paths()
    train_data, test_data, threshold = get_dataset(task_type)
    string_data_input = check_data_type(ml_task=Task(TaskTypesEnum.regression),
                                        features=path_to_train)
    array_data_input = check_data_type(ml_task=Task(TaskTypesEnum.regression),
                                       features=x_train)
    fedot_data_input = check_data_type(ml_task=Task(TaskTypesEnum.regression),
                                       features=train_data)
    assert not type(string_data_input) == InputData \
           or type(array_data_input) == InputData \
           or type(fedot_data_input) == InputData
