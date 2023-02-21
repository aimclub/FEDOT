from functools import partial

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
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from test.unit.pipelines.test_decompose_pipelines import get_classification_data
from test.unit.tasks.test_forecasting import get_ts_data

TABULAR_SIMPLE = {'train_features_size': (8, 5), 'test_features_size': (2, 5), 'test_idx': (8, 9)}
TS_SIMPLE = {'train_features_size': (18,), 'test_features_size': (18,), 'test_idx': (18, 19)}
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
    ts = np.arange(0, 20)
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


def get_imbalanced_data_to_test_mismatch():
    task = Task(TaskTypesEnum.classification)
    x = np.array([[0, 0, 15],
                 [0, 1, 2],
                 [8, 12, 0],
                 [0, 1, 0],
                 [1, 1, 0],
                 [0, 11, 9],
                 [5, 1, 10],
                 [8, 16, 4],
                 [3, 1, 5],
                 [0, 1, 6],
                 [2, 7, 9],
                 [0, 1, 2],
                 [14, 1, 0],
                 [0, 4, 10]])
    y = np.array([0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 0, 3, 3])
    input_data = InputData(idx=np.arange(0, len(x)), features=x,
                           target=y, task=task, data_type=DataTypesEnum.table)
    return input_data


def get_balanced_data_to_test_mismatch():
    task = Task(TaskTypesEnum.classification)
    x = np.array([[0, 0, 15],
                 [0, 1, 2],
                 [8, 12, 0],
                 [0, 1, 0],
                 [1, 1, 0],
                 [0, 11, 9],
                 [5, 1, 10],
                 [8, 16, 4],
                 [3, 1, 5],
                 [0, 1, 6],
                 [2, 7, 9],
                 [0, 1, 2],
                 [14, 1, 0],
                 [0, 4, 10]])
    y = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 1, 0, 0, 3, 3])
    input_data = InputData(idx=np.arange(0, len(x)), features=x,
                           target=y, task=task, data_type=DataTypesEnum.table)
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

    assert len(train_data.features) == len(train_data.target) == 16
    # For in sample validation it is required to have all length of time series
    assert len(test_data.features) == 20
    assert np.allclose(test_data.target, np.array([16, 17, 18, 19]))


@pytest.mark.parametrize('data_splitter, data',
                         # test StratifiedKFold
                         [(DataSourceSplitter(cv_folds=3, shuffle=True), get_imbalanced_data_to_test_mismatch()),
                          # test KFold
                          (DataSourceSplitter(cv_folds=3, shuffle=True), get_balanced_data_to_test_mismatch()),
                          # test hold-out
                          (DataSourceSplitter(shuffle=True), get_imbalanced_data_to_test_mismatch())])
def test_data_splitting_without_shape_mismatch(data_splitter: DataSourceSplitter, data: InputData):
    """ Checks if data split correctly into train test subsets: there are no new classes in test subset """
    data_source = data_splitter.build(data=data)
    for fold_id, (train_data, test_data) in enumerate(data_source()):
        assert set(train_data.target) >= set(test_data.target)


def test_data_splitting_perform_correctly_after_build():
    """
    Check if data splitting perform correctly through Objective Builder - Objective Evaluate
    Case: in-sample validation during cross validation for time series forecasting
    """
    output_by_fold = {0: {'train_features_size': (12,), 'test_features_size': (16,), 'test_target_size': (16,)},
                      1: {'train_features_size': (16,), 'test_features_size': (20,), 'test_target_size': (20,)}}

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
        assert len(train_series_data.features) == len(train_series_data.target) == 16

    for series_id, test_series_data in test_data.items():
        assert len(test_series_data.features) == 20
        assert np.allclose(test_series_data.target, np.array([16, 17, 18, 19]))\



@pytest.mark.parametrize("cv_generator, data",
                         [(partial(tabular_cv_generator, folds=5),
                           get_classification_data()[0]),
                          (partial(ts_cv_generator, folds=3, validation_blocks=2),
                           get_ts_data()[0])])
def test_cv_generator_works_stable(cv_generator, data):
    """ Test if ts cv generator works stable (always return same folds) """
    idx_first = []
    idx_second = []
    for row in cv_generator(data=data):
        idx_first.append(row[1].idx)
    for row in cv_generator(data=data):
        idx_second.append(row[1].idx)

    for i in range(len(idx_first)):
        assert np.all(idx_first[i] == idx_second[i])
