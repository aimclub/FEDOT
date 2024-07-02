import os
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root, split_data
from fedot.core.data.cv_folds import cv_generator
from test.unit.pipelines.test_decompose_pipelines import get_classification_data
from test.unit.tasks.test_forecasting import get_ts_data

TABULAR_SIMPLE = {'train_features_size': (8, 5), 'test_features_size': (2, 5), 'test_idx': (8, 9)}
TS_SIMPLE = {'train_features_size': (18,), 'test_features_size': (18,), 'test_idx': (18, 19)}
TEXT_SIMPLE = {'train_features_size': (8,), 'test_features_size': (2,), 'test_idx': (8, 9)}
IMAGE_SIMPLE = {'train_features_size': (8, 5, 5, 2), 'test_features_size': (2, 5, 5, 2), 'test_idx': (8, 9)}


def get_tabular_classification_data(length=10, class_count=2):
    task = Task(TaskTypesEnum.classification)
    features = np.full((length, 5), 1, dtype=float)
    target = np.repeat(np.array(list(range(1, class_count + 1))), length // class_count).reshape((-1, 1))
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=target, task=task, data_type=DataTypesEnum.table)
    return input_data


def get_ts_data_to_forecast(forecast_length, data_shape=100):
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length))
    ts = np.arange(0, data_shape)
    input_data = InputData(idx=range(0, len(ts)), features=ts,
                           target=ts, task=task, data_type=DataTypesEnum.ts)
    return input_data


def get_ts_data_to_forecast_two_elements():
    return get_ts_data_to_forecast(2, 20)


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


def check_shuffle(sample):
    unique = np.unique(np.diff(sample.idx))
    test_result = len(unique) > 1 or np.min(unique) > 1
    return test_result


def check_stratify(train, test):
    deltas = [np.unique(np.sort(train.target), return_counts=True)[1],
              np.unique(np.sort(test.target), return_counts=True)[1]]
    return np.allclose(*[delta / sum(delta) for delta in deltas])


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
    train_data, test_data = train_test_data_setup(input_data, stratify=False)

    assert train_data.features.shape == expected_output['train_features_size']
    assert test_data.features.shape == expected_output['test_features_size']
    assert tuple(test_data.idx) == expected_output['test_idx']


def test_multitarget_train_test_split():
    """ Checks multitarget stratification for dataset with unbalanced distribution of classes """
    target_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    full_test_data_path = os.path.join(str(fedot_project_root()), 'test', 'data', 'multitarget_classification.csv')
    data = InputData.from_csv(
        file_path=full_test_data_path, task="classification", target_columns=target_columns, columns_to_drop=["id"]
    )
    train, test = train_test_data_setup(data)


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
                         [(DataSourceSplitter(cv_folds=3, shuffle=True), get_imbalanced_data_to_test_mismatch()),
                          (DataSourceSplitter(cv_folds=3, shuffle=True), get_balanced_data_to_test_mismatch()),
                          (DataSourceSplitter(shuffle=True), get_imbalanced_data_to_test_mismatch()),
                          ])
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
        assert np.allclose(test_series_data.target, np.array([16, 17, 18, 19]))


@pytest.mark.parametrize(
    ("datas_funs", "cv_folds", "shuffle", "stratify"),
    [
        # classification + stratify + shuffle + cv_folds
        ([partial(get_tabular_classification_data, 100, 5)] * 3, 4, True, True),
        # classification + shuffle + cv_folds
        ([partial(get_tabular_classification_data, 100, 5)] * 3, 4, True, False),
        # classification + cv_folds
        ([partial(get_tabular_classification_data, 100, 5)] * 3, 4, False, False),
        # classification + stratify + shuffle
        ([partial(get_tabular_classification_data, 100, 5)] * 3, None, True, True),
        # classification + shuffle
        ([partial(get_tabular_classification_data, 100, 5)] * 3, None, True, False),
        # classification
        ([partial(get_tabular_classification_data, 100, 5)] * 3, None, False, False),
        # timeseries + cv_folds
        ([partial(get_ts_data_to_forecast, 10, 100)] * 3, 3, False, False),
        # timeseries
        ([partial(get_ts_data_to_forecast, 10, 100)] * 3, None, False, False),
    ],
)
def test_multimodal_data_splitting_is_correct(datas_funs, cv_folds, shuffle, stratify):
    mdata = MultiModalData({f'data_{i}': data_fun() for i, data_fun in enumerate(datas_funs)})
    data_splitter = DataSourceSplitter(cv_folds=cv_folds, shuffle=shuffle, stratify=stratify)
    data_producer = data_splitter.build(mdata)
    keys = tuple(mdata.keys())
    features_dimensity = [subdata.features.shape[1:] for subdata in mdata.values()]

    for samples in data_producer():
        for sample in samples:
            assert isinstance(sample, MultiModalData)

            # keys should be the same
            assert set(keys) == set(sample.keys())

            # idx should be the same
            idx = [np.reshape(x.idx, (-1, 1)) for x in sample.values()]
            assert np.all(np.diff(np.concatenate(idx, 1), 1) == 0)

            # dimensity of features should be the same
            splitted_data_features_dimensity = [subdata.features.shape[1:] for subdata in sample.values()]
            assert features_dimensity == splitted_data_features_dimensity

            # shuffle should be done
            if shuffle:
                for key in keys:
                    assert check_shuffle(sample[key])

        # stratify should be done
        if stratify:
            for key in keys:
                assert check_stratify(samples[0][key], samples[1][key])


@pytest.mark.parametrize("cv_generator, data",
                         [(partial(cv_generator, cv_folds=5),
                           get_classification_data()[0]),
                          (partial(cv_generator, cv_folds=3, validation_blocks=2),
                           get_ts_data()[0])])
def test_cv_generator_works_stable(cv_generator, data):
    """ Test if ts cv generator works stable (always return same folds) """
    idx_first = []
    idx_second = []
    for row in cv_generator(data=data, stratify=False, random_seed=None):
        idx_first.append(row[1].idx)
    for row in cv_generator(data=data, stratify=False, random_seed=None):
        idx_second.append(row[1].idx)

    for i in range(len(idx_first)):
        assert np.all(idx_first[i] == idx_second[i])


@pytest.mark.parametrize(("forecast_length, cv_folds, split_ratio,"
                          "check_cv_folds, check_split_ratio, check_validation_blocks"),
                         [(30, 3, 0.5, 2, 0.5, 1),
                          (50, 3, 0.5, None, 0.5, 1),
                          (60, None, 0.5, None, 0.4, 1),
                          (10, 3, 0.5, 3, 0.5, 2),
                          (10, 3, 0.6, 3, 0.6, 2),
                          (20, 3, 0.5, 3, 0.5, 1),
                          (25, 3, 0.5, 3, 0.5, 1),
                          (5, 3, 0.5, 3, 0.5, 5)])
def test_data_splitting_defines_validation_blocks_correctly(forecast_length, cv_folds, split_ratio,
                                                            check_cv_folds, check_split_ratio,
                                                            check_validation_blocks):
    """ Checks if validation blocks count defines correctly for different data """
    data = get_ts_data_to_forecast(forecast_length, 120)
    data_source_splitter = DataSourceSplitter(cv_folds=cv_folds, split_ratio=split_ratio)
    data_source_splitter.build(data)
    assert data_source_splitter.cv_folds == check_cv_folds
    assert data_source_splitter.split_ratio == check_split_ratio
    assert data_source_splitter.validation_blocks == check_validation_blocks


@pytest.mark.parametrize(('cv_folds', 'shuffle', 'stratify', 'data_classes'),
                         [(2, True, True, 2),  # simple case
                          (2, False, True, 2),  # should work without error
                          (5, True, True, 4),  # more folds and more classes
                          ])
def test_stratify(cv_folds, shuffle, stratify, data_classes):
    data = get_tabular_classification_data(length=100, class_count=data_classes)
    data_splitter = DataSourceSplitter(cv_folds=cv_folds, shuffle=shuffle, stratify=stratify)
    data_producer = data_splitter.build(data)

    for train, test in data_producer():
        assert check_stratify(train, test)


@pytest.mark.parametrize(('is_shuffle', 'shuffle', 'cv_folds', 'data'),
                         [(True, True, 2,
                           get_tabular_classification_data(length=100, class_count=4)),  # cv_folds classification
                          (True, True, None,
                           get_tabular_classification_data(length=100, class_count=4)),  # holdout classification
                          (False, True, 2, get_ts_data_to_forecast(10, 100)),  # cv_folds timeseries
                          (False, True, None, get_ts_data_to_forecast(10, 100)),  # holdout timeseries
                          ])
def test_shuffle(is_shuffle, cv_folds, shuffle, data):
    data_splitter = DataSourceSplitter(cv_folds=cv_folds, shuffle=shuffle, stratify=False)
    data_producer = data_splitter.build(data)

    for samples in data_producer():
        for sample in samples:
            assert check_shuffle(sample) == is_shuffle
