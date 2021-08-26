import os
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, \
    make_classification, make_regression, make_gaussian_quantiles

from fedot.core.utils import fedot_project_root
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from examples.image_classification_problem import run_image_classification_problem
from cases.data.data_utils import get_scoring_case_data_paths
from cases.credit_scoring.credit_scoring_problem import get_scoring_data


def get_data(task):
    full_path_train, full_path_test = get_scoring_data()
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    dataset_to_validate = InputData.from_csv(full_path_test, task=task)

    return dataset_to_compose, dataset_to_validate


def generate_gaps_in_ts(array_without_gaps, gap_dict, gap_value):
    """
    Function for generating gaps with predefined length in the desired indices
    of an one-dimensional array (time series)

    :param array_without_gaps: an array without gaps
    :param gap_dict: a dictionary with omissions, where the key is the index in
    the time series from which the gap will begin. The key value is the length
    of the gap (elements). -1 in the value means that a skip is generated until
    the end of the array
    :param gap_value: value indicating a gap in the array

    :return: one-dimensional array with omissions
    """

    array_with_gaps = np.copy(array_without_gaps)

    keys = list(gap_dict.keys())
    for key in keys:
        gap_size = gap_dict.get(key)
        if gap_size == -1:
            # Generating a gap to the end of an array
            array_with_gaps[key:] = gap_value
        else:
            array_with_gaps[key:(key + gap_size)] = gap_value

    return array_with_gaps


def get_array_with_gaps(gap_dict=None, gap_value: float = -100.0):
    """
    Function for generating synthetic data and gaps in it with predefined length
    and location

    :param gap_dict: a dictionary with omissions, where the key is the index in
    the time series from which the gap will begin. The key value is the length
    of the gap (elements). -1 in the value means that a skip is generated until
    the end of the array
    :param gap_value: value indicating a gap in the array

    :return array_with_gaps: an array with gaps
    :return real_values: an array with actual values in gaps
    """

    real_values = generate_synthetic_data()

    if gap_dict is None:
        gap_dict = {850: 100,
                    1400: 150}
    array_with_gaps = generate_gaps_in_ts(array_without_gaps=real_values,
                                          gap_dict=gap_dict,
                                          gap_value=gap_value)

    return array_with_gaps, real_values


def get_classification_data(classes_amount: int):
    """ Function generate synthetic dataset for classification task

    :param classes_amount: amount of classes to predict

    :return train_input: InputData for model fit
    :return predict_input: InputData for predict stage
    """

    # Define options for dataset with 800 objects
    features_options = {'informative': 2, 'redundant': 1,
                        'repeated': 1, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                  800, 4,
                                                                  classes_amount)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Define classification task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train and validate the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train, target=y_train,
                            task=task, data_type=DataTypesEnum.table)
    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test, target=y_test,
                              task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input


def classification_synthetic(samples_amount: int, features_amount: int, classes_amount: int,
                             features_options: Dict, noise_fraction: float = 0.1,
                             full_shuffle: bool = True):
    """
    Generates a random dataset for n-class classification problem
    using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value \
        format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :param noise_fraction: the fraction of noisy labels in the dataset;
    :param full_shuffle: if true then all features and samples will be shuffled.
    :return: features and target as numpy-arrays.
    """
    features, target = make_classification(n_samples=samples_amount, n_features=features_amount,
                                           n_informative=features_options['informative'],
                                           n_redundant=features_options['redundant'],
                                           n_repeated=features_options['repeated'],
                                           n_classes=classes_amount,
                                           n_clusters_per_class=features_options['clusters_per_class'],
                                           flip_y=noise_fraction,
                                           shuffle=full_shuffle)

    return features, target


def regression_synthetic(samples_amount: int, features_amount: int, features_options: Dict,
                         n_targets: int, noise: float = 0.0, shuffle: bool = True):
    """
    Generates a random dataset for regression problem using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value \
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :param n_targets: the amount of target variables;
    :param noise: the standard deviation of the gaussian noise applied to the output;
    :param shuffle: if true then all features and samples will be shuffled.
    :return: features and target as numpy-arrays.
    """

    features, target = make_regression(n_samples=samples_amount, n_features=features_amount,
                                       n_informative=features_options['informative'],
                                       bias=features_options['bias'],
                                       n_targets=n_targets,
                                       noise=noise,
                                       shuffle=shuffle)

    return features, target


def gauss_quantiles_synthetic(samples_amount: int, features_amount: int,
                              classes_amount: int, full_shuffle=True, **kwargs):
    """
    Generates a random dataset for n-class classification problem
    based on multi-dimensional gaussian distribution quantiles
    using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param full_shuffle: if true then all features and samples will be shuffled.
    :param kwargs: Optional params: \
        - 'gauss_params': mean and covariance values of the distribution.
    :return: features and target as numpy-arrays.
    """
    if 'gauss_params' in kwargs:
        mean, cov = kwargs['gauss_params']
    else:
        mean, cov = None, 1.

    features, target = make_gaussian_quantiles(n_samples=samples_amount,
                                               n_features=features_amount,
                                               n_classes=classes_amount,
                                               shuffle=full_shuffle,
                                               mean=mean, cov=cov)
    return features, target


def generate_synthetic_data(length: int = 2200, periods: int = 5):
    """
    The function generates a synthetic one-dimensional array without omissions

    :param length: the length of the array
    :param periods: the number of periods in the sine wave

    :return synthetic_data: an array without gaps
    """

    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise
    return synthetic_data


def get_regression_dataset(features_options, samples_amount=250,
                           features_amount=5):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.3)

    return x_train, y_train, x_test, y_test


def get_classification_dataset(features_options, samples_amount=250,
                               features_amount=5, classes_amount=2):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=features_amount,
                                            classes_amount=classes_amount,
                                            features_options=features_options)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data,
                                                                            test_size=0.3)

    return x_data_train, y_data_train, x_data_test, y_data_test


def file_data():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def classification_dataset_with_redunant_features(
        n_samples=1000, n_features=100, n_informative=5) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative)

    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def create_data_for_train():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


def synthetic_univariate_ts(forecast_length):
    """ Method returns InputData for classical time series forecasting task """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    # Simple time series to process
    ts_train = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    ts_test = np.array([140, 150, 160, 170])

    # Prepare train data
    train_input = InputData(idx=np.arange(0, len(ts_train)),
                            features=ts_train,
                            target=ts_train,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(ts_train)
    end_forecast = start_forecast + forecast_length
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=ts_train,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)
    return train_input, predict_input, ts_test


def synthetic_with_exogenous_ts(forecast_length):
    """ Method returns InputData for time series forecasting task with
    exogenous variable """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # Time series with exogenous variable
    ts_train = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    ts_exog = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

    ts_test = np.array([140, 150, 160, 170])
    ts_test_exog = np.array([24, 25, 26, 27])

    # Indices for forecast
    start_forecast = len(ts_train)
    end_forecast = start_forecast + forecast_length

    # Input for source time series
    train_source_ts = InputData(idx=np.arange(0, len(ts_train)),
                                features=ts_train, target=ts_train,
                                task=task, data_type=DataTypesEnum.ts)
    predict_source_ts = InputData(idx=np.arange(start_forecast, end_forecast),
                                  features=ts_train, target=None,
                                  task=task, data_type=DataTypesEnum.ts)

    # Input for exogenous variable
    train_exog_ts = InputData(idx=np.arange(0, len(ts_train)),
                              features=ts_exog, target=ts_train,
                              task=task, data_type=DataTypesEnum.ts)
    predict_exog_ts = InputData(idx=np.arange(start_forecast, end_forecast),
                                features=ts_test_exog, target=None,
                                task=task, data_type=DataTypesEnum.ts)
    return train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test


def text_input_data():
    test_text = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    input_data = InputData(features=test_text,
                           target=[0, 1, 1, 0],
                           idx=np.arange(0, len(test_text)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.text)
    return input_data


def get_small_regression_dataset():
    """ Function returns features and target for train and test regression models """
    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=70,
                                                              features_amount=4)
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_small_classification_dataset():
    """ Function returns features and target for train and test classification models """
    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options=features_options,
                                                                  samples_amount=70,
                                                                  features_amount=4,
                                                                  classes_amount=2)
    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_time_series():
    """ Function returns time series for time series forecasting task """
    len_forecast = 100
    synthetic_ts = generate_synthetic_data(length=1000)

    train_data = synthetic_ts[:-len_forecast]
    test_data = synthetic_ts[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, test_data


def get_nan_inf_data():
    train_input = InputData(idx=[0, 1, 2, 3],
                            features=np.array([[1, 2, 3, 4],
                                               [2, np.nan, 4, 5],
                                               [3, 4, 5, np.inf],
                                               [-np.inf, 5, 6, 7]]),
                            target=np.array([1, 2, 3, 4]),
                            task=Task(TaskTypesEnum.regression),
                            data_type=DataTypesEnum.table)

    return train_input


def output_dataset():
    task = Task(TaskTypesEnum.classification)

    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = OutputData(idx=np.arange(0, 100), features=x, predict=classes,
                      task=task, data_type=DataTypesEnum.table)

    return data


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


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/advanced_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


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
