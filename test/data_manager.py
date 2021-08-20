import os
import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, make_classification, make_regression

from fedot.core.utils import fedot_project_root
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from examples.image_classification_problem import run_image_classification_problem


def get_synthetic_regression_data(n_samples=1000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.regression),
                           data_type=DataTypesEnum.table)
    return input_data


def get_iris_data() -> InputData:
    synthetic_data = load_iris()
    input_data = InputData(idx=np.arange(0, len(synthetic_data.target)),
                           features=synthetic_data.data,
                           target=synthetic_data.target,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def get_binary_classification_data():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    return input_data


def get_image_classification_data(composite_flag: bool = True):
    """ Method for loading data with images in .npy format (training_data.npy, training_labels.npy,
    test_data.npy, test_labels.npy) that are used in tests.This npy files are a truncated version
    of the MNIST dataset, that contains only 10 first images.

    :param composite_flag: Flag that allows to run tests for complex composite models
    """
    test_data_path = '../../data/test_data.npy'
    test_labels_path = '../../data/test_labels.npy'
    train_data_path = '../../data/training_data.npy'
    train_labels_path = '../../data/training_labels.npy'

    test_file_path = str(os.path.dirname(__file__))
    training_path_features = os.path.join(test_file_path, train_data_path)
    training_path_labels = os.path.join(test_file_path, train_labels_path)
    test_path_features = os.path.join(test_file_path, test_data_path)
    test_path_labels = os.path.join(test_file_path, test_labels_path)

    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
        train_dataset=(training_path_features,
                       training_path_labels),
        test_dataset=(test_path_features,
                      test_path_labels),
        composite_flag=composite_flag)

    return roc_auc_on_valid, dataset_to_train, dataset_to_validate


def get_synthetic_input_data(n_samples=10000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def get_ts_data(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'test/data/simple_time_series.csv')
    df = pd.read_csv(file_path)

    time_series = np.array(df['sea_height'])[:n_steps]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data)


def import_metoocean_data():
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    target_history, add_history, _ = prepare_input_data(full_path_train, full_path_test)


def get_split_data_paths():
    file_path_train = 'data/simple_regression_train.csv'
    file_path_test = 'data/simple_regression_test.csv'
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
        data = get_synthetic_input_data(n_samples=1000)
        train_data, test_data = train_test_data_setup(data)
        threshold = 0.5
    elif task_type == 'ts_forecasting':
        train_data, test_data = get_ts_data(forecast_length=5)
        threshold = np.std(test_data.target)
    else:
        raise ValueError('Incorrect type of machine learning task')
    return train_data, test_data, threshold


@pytest.fixture()
def file_data_setup():
    def _to_numerical(categorical_ids: np.ndarray):
        encoded = pd.factorize(categorical_ids)[0]
        return encoded

    test_file_path = str(os.path.dirname(__file__))
    file = 'data/advanced_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]

    # Wrap data into InputData
    input_data = InputData(features=predictors,
                           target=response,
                           idx=np.arange(0, len(predictors)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    # Train test split
    train_data, test_data = train_test_data_setup(input_data)
    return train_data, test_data


def multi_target_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/multi_target_sample.csv'
    path = os.path.join(test_file_path, file)

    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    task = Task(TaskTypesEnum.regression)
    data = InputData.from_csv(path, target_columns=target_columns,
                              columns_to_drop=['date'], task=task)
    train, test = train_test_data_setup(data)
    return train, test


def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_regression.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))

