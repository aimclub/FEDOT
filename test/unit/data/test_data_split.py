import pandas as pd
import numpy as np
import pytest
from typing import Callable

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import split_data

TABULAR_SIMPLE = {'train_features_size': (8, 5), 'test_features_size': (2, 5), 'test_idx': (8, 9)}
TS_SIMPLE = {'train_features_size': (8,), 'test_features_size': (8,), 'test_idx': (8, 9)}
TEXT_SIMPLE = {'train_features_size': (8,), 'test_features_size': (2,), 'test_idx': (8, 9)}
IMAGE_SIMPLE = {'train_features_size': (8, 5, 5, 2), 'test_features_size': (2, 5, 5, 2), 'test_idx': (8, 9)}


def get_tabular_classification_data():
    task = Task(TaskTypesEnum.classification)
    features = np.full((10, 5), 1, dtype=float)
    target = np.repeat(np.array([1, 2]), 5).reshape((-1, 1))
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=target, task=task, data_type=DataTypesEnum.table)
    return input_data


def get_ts_data_to_forecast_two_elements():
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2))
    ts = np.arange(0, 10)
    input_data = InputData(idx=range(0, len(ts)), features=ts,
                           target=ts, task=task, data_type=DataTypesEnum.ts)
    return input_data


def get_text_classification_data():
    task = Task(TaskTypesEnum.classification)
    features = np.array(['blue', 'da', 'bu', 'di', 'da', 'bu', 'dai', 'I am blue', 'da', 'bu'])
    target = np.full((10, 1), 1, dtype=float)
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=target, task=task, data_type=DataTypesEnum.text)
    return input_data


def get_image_classification_data():
    task = Task(TaskTypesEnum.classification)
    # Imitate small pictures (5 x 5) with this array
    features = np.random.random((10, 5, 5, 2))
    target = np.full((10, 1), 1, dtype=float)
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=target, task=task, data_type=DataTypesEnum.image)
    return input_data


def test_split_data():
    dataframe = pd.DataFrame(data=[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12],
                                   [13, 14, 15]])
    train, test = split_data(dataframe)

    assert len(train) == 4
    assert len(test) == 1


@pytest.mark.parametrize('data_generator, expected_output',
                         [(get_tabular_classification_data, TABULAR_SIMPLE),
                          (get_ts_data_to_forecast_two_elements, TS_SIMPLE),
                          (get_text_classification_data, TEXT_SIMPLE),
                          (get_image_classification_data, IMAGE_SIMPLE)])
def test_default_train_test_simple(data_generator: Callable, expected_output: dict):
    """ Check if simple splitting perform correctly for all used in FEDOT data types """
    input_data = data_generator()
    train_data, test_data = train_test_data_setup(input_data)

    assert train_data.features.shape == expected_output['train_features_size']
    assert test_data.features.shape == expected_output['test_features_size']
    assert tuple(test_data.idx) == expected_output['test_idx']


def test_advanced_time_series_splitting():
    """ Check how data prepared for time series in-sample validation """
    validation_blocks = 2
    time_series = get_ts_data_to_forecast_two_elements()
    train_data, test_data = train_test_data_setup(time_series, validation_blocks=validation_blocks)

    assert len(train_data.features) == len(train_data.target) == 6
    # For in sample validation it is required to have all length of time series
    assert len(test_data.features) == 10
    assert np.allclose(test_data.target, np.array([6, 7, 8, 9]))


def test_data_splitting_perform_correctly_after_build():
    """
    Check if data splitting perform correctly through Objective Builder - Objective Evaluate
    Case: in-sample validation during cross validation for time series forecasting
    """
    output_by_fold = {0: {'train_features_size': (2,), 'test_features_size': (6,), 'test_target_size': (6,)},
                      1: {'train_features_size': (6,), 'test_features_size': (10,), 'test_target_size': (10,)}}

    time_series = get_ts_data_to_forecast_two_elements()
    data_source = DataSourceSplitter(cv_folds=2, validation_blocks=2).build(time_series)

    # Imitate evaluation process
    for fold_id, (train_data, test_data) in enumerate(data_source()):

        expected_output = output_by_fold[fold_id]
        assert train_data.features.shape == expected_output['train_features_size']
        assert test_data.features.shape == expected_output['test_features_size']
        assert test_data.target.shape == expected_output['test_target_size']


def test_multivariate_time_series_splitting_correct():
    """ Check if in-sample for multivariate time series work correctly """
    multivar_ts = MultiModalData({'series_1': get_ts_data_to_forecast_two_elements(),
                                  'series_2': get_ts_data_to_forecast_two_elements()})

    train_data, test_data = train_test_data_setup(multivar_ts, validation_blocks=2)
    for series_id, train_series_data in train_data.items():
        assert len(train_series_data.features) == len(train_series_data.target) == 6

    for series_id, test_series_data in test_data.items():
        assert len(test_series_data.features) == 10
        assert np.allclose(test_series_data.target, np.array([6, 7, 8, 9]))
