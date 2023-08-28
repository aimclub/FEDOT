import os
from copy import deepcopy, copy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from fedot.core.data.data import InputData, get_df_from_csv
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.unit.tasks.test_forecasting import get_ts_data_with_dt_idx


@pytest.fixture()
def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


@pytest.fixture()
def ts_setup() -> InputData:
    data = -np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data = InputData(features=data, target=data, idx=np.arange(len(data)),
                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(2)),
                     data_type=DataTypesEnum.ts)
    return data


def check_shuffle_order(data, shuffled_data, reorder_map=None):
    if reorder_map is None:
        shuffled_data_idx = list(shuffled_data.idx)
        reorder_map = dict()
        for i, val in enumerate(data.idx):
            if val in shuffled_data_idx:
                reorder_map[i] = shuffled_data_idx.index(val)
        assert len(reorder_map) == len(shuffled_data_idx)

    if isinstance(data, InputData):
        assert all(np.array_equal(data.features[i], shuffled_data.features[reorder_map[i]]) for i in reorder_map)
        assert all(data.target[i] == shuffled_data.target[reorder_map[i]] for i in reorder_map)
    else:
        for key in data.keys():
            check_shuffle_order(data[key], shuffled_data[key], reorder_map)


def test_data_subset_correct(data_setup):
    subset_size = (10, 50)
    subset_length = subset_size[1] - subset_size[0]
    subset = data_setup.subset_range(subset_size[0], subset_size[1] - 1)

    assert len(subset.idx) == subset_length
    assert np.array_equal(subset.idx, data_setup.idx[subset_size[0]:subset_size[1]])
    assert len(subset.features) == subset_length
    assert np.array_equal(subset.features, data_setup.features[subset_size[0]:subset_size[1]])
    assert len(subset.target) == subset_length
    assert np.array_equal(subset.target, data_setup.target[subset_size[0]:subset_size[1]])
    assert subset.task == data_setup.task
    assert subset.data_type == data_setup.data_type


def test_data_subset_for_ts_correct(ts_setup):
    _, test = train_test_data_setup(ts_setup, validation_blocks=2)
    subset_size = (1, 3)
    subset = test.subset_range(subset_size[0], subset_size[1] - 1)
    assert np.array_equal(subset.idx, test.idx[subset_size[0]:subset_size[1]])
    assert np.array_equal(subset.features, test.features[:-(test.idx.shape[0] - subset_size[1])])
    assert np.array_equal(subset.target, test.target[subset_size[0]:subset_size[1]])
    assert subset.task == test.task
    assert subset.data_type == test.data_type


def test_data_slice(data_setup):
    for islice in ((None, 5, None), (None, -5, None), (2, 5, None), (2, 8, 2), (-6, -2, None), (-8, -2, 2)):
        sliced = data_setup.slice(*islice)
        assert np.array_equal(data_setup.idx[slice(*islice)], sliced.idx)
        assert np.array_equal(data_setup.features[slice(*islice)], sliced.features)
        assert np.array_equal(data_setup.target[slice(*islice)], sliced.target)


def test_ts_slice(ts_setup):# check common slice
    for islice in ((None, 5, 1), (None, -5, 1), (2, 5, 1), (2, 8, 2), (-6, -2, 1)):
        sliced = ts_setup.slice(*islice)
        assert np.array_equal(ts_setup.idx[slice(*islice)], sliced.idx)

        features_slice = slice(0, islice[1], islice[2])
        assert np.array_equal(ts_setup.features[features_slice], sliced.features)

        sliced_target = ts_setup.target[slice(*islice)]
        assert np.array_equal(sliced_target, sliced.features[-len(sliced_target):])
        assert np.array_equal(sliced_target, sliced.target)

    islice = (-8, -2, 2)
    sliced = ts_setup.slice(*islice)
    assert np.array_equal(ts_setup.idx[slice(*islice)], sliced.idx)
    assert np.array_equal([-1, -3, -5, -7], sliced.features)
    assert np.array_equal(ts_setup.target[slice(*islice)], sliced.target)


def test_ts_data_slice_for_partial_ts(ts_setup):
    _, ts_setup = train_test_data_setup(ts_setup, validation_blocks=3)
    delta = ts_setup.features.shape[0] - ts_setup.idx.shape[0]
    for islice in ((None, 2, 1), (None, -2, 1), (1, 3, 1), (1, 4, 2), (-3, -1, 1)):
        sliced = ts_setup.slice(*islice)
        assert np.array_equal(ts_setup.idx[slice(*islice)], sliced.idx)

        if islice[1] > 0:
            features_slice = slice(0, islice[1] + delta, islice[2])
            assert np.array_equal(ts_setup.features[features_slice], sliced.features)
        elif islice[0] is None:
            assert np.array_equal(ts_setup.features[slice(*islice)], sliced.features)
        else:
            features_slice = slice(0, islice[1] , islice[2])
            assert np.array_equal(ts_setup.features[features_slice], sliced.features)

        sliced_target = ts_setup.target[slice(*islice)]
        assert np.array_equal(sliced_target, sliced.features[-len(sliced_target):])
        assert np.array_equal(sliced_target, sliced.target)

    islice = (-4, -1, 2)
    sliced = ts_setup.slice(*islice)
    assert np.array_equal(ts_setup.idx[slice(*islice)], sliced.idx)
    assert np.array_equal([-1, -3, -5, -7, -9], sliced.features)
    assert np.array_equal(ts_setup.target[slice(*islice)], sliced.target)


def test_ts_slice_mirrowed(ts_setup):
    for data1, data2 in zip((ts_setup.slice(None, -1),
                             ts_setup.slice(None, -2)),
                            (ts_setup.slice(None, len(ts_setup) - 1),
                             ts_setup.slice(None, len(ts_setup) - 2))):
        assert np.array_equal(data1.idx, data2.idx)
        assert np.array_equal(data1.features, data2.features)
        assert np.array_equal(data1.target, data2.target)


def test_data_subset_incorrect(data_setup):
    subset_size = 105
    with pytest.raises(IndexError):
        assert data_setup.subset_range(0, subset_size)
    with pytest.raises(IndexError):
        assert data_setup.subset_range(-1, subset_size)
    with pytest.raises(IndexError):
        assert data_setup.subset_range(-1, -1)
    with pytest.raises(IndexError):
        assert data_setup.subset_range(10, 2)


def test_data_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    task = Task(TaskTypesEnum.classification)
    df = pd.read_csv(os.path.join(test_file_path, file))
    data_array = np.array(df).T
    features = data_array[1:-1].T
    target = data_array[-1]
    idx = data_array[0]
    expected_features = InputData(features=features, target=target,
                                  idx=idx,
                                  task=task,
                                  data_type=DataTypesEnum.table).features
    actual_features = InputData.from_csv(
        os.path.join(test_file_path, file)).features
    assert np.array_equal(expected_features, actual_features)


def test_with_custom_target():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    file_custom = '../../data/simple_classification_with_custom_target.csv'

    file_data = InputData.from_csv(
        os.path.join(test_file_path, file))

    expected_features = file_data.features
    expected_target = file_data.target

    custom_file_data = InputData.from_csv(
        os.path.join(test_file_path, file_custom), delimiter=';')
    actual_features = custom_file_data.features
    actual_target = custom_file_data.target

    assert not np.array_equal(expected_features, actual_features)
    assert not np.array_equal(expected_target, actual_target)

    custom_file_data = InputData.from_csv(
        os.path.join(test_file_path, file_custom), delimiter=';',
        columns_to_drop=['redundant'], target_columns='custom_target')

    actual_features = custom_file_data.features
    actual_target = custom_file_data.target

    assert np.array_equal(expected_features, actual_features)
    assert np.array_equal(expected_target, actual_target)


def test_data_from_json():
    # several features
    files_path = os.path.join('test', 'data', 'multi_modal')
    path = os.path.join(str(fedot_project_root()), files_path)
    data = InputData.from_json_files(path, fields_to_use=['votes', 'year'],
                                     label='rating', task=Task(TaskTypesEnum.regression))
    assert data.features.shape[1] == 2  # check there is two features
    assert len(data.target) == data.features.shape[0] == len(data.idx)

    # single feature
    data = InputData.from_json_files(path, fields_to_use=['votes'],
                                     label='rating', task=Task(TaskTypesEnum.regression))
    assert len(data.features.shape) == 1  # check there is one feature
    assert len(data.target) == len(data.features) == len(data.idx)


def test_target_data_from_csv_correct():
    """ Function tests two ways of processing target columns in "from_csv"
    method
    """
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/multi_target_sample.csv'
    path = os.path.join(test_file_path, file)
    task = Task(TaskTypesEnum.regression)

    # Process one column
    target_column = '1_day'
    one_column_data = InputData.from_csv(path, target_columns=target_column,
                                         index_col='date', task=task)

    # Process multiple target columns
    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    seven_columns_data = InputData.from_csv(path, target_columns=target_columns,
                                            index_col='date', task=task)

    assert one_column_data.target.shape == (197, 1)
    assert seven_columns_data.target.shape == (197, 7)


def test_table_data_shuffle():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'

    data = InputData.from_csv(os.path.join(test_file_path, file))
    shuffled_data = deepcopy(data)
    shuffled_data.shuffle()

    check_shuffle_order(data, shuffled_data)

    assert not np.array_equal(data.idx, shuffled_data.idx)
    assert not np.array_equal(data.features, shuffled_data.features)
    assert not np.array_equal(data.target, shuffled_data.target)


def test_data_convert_string_indexes_correct():
    """ Test is string indexes converted correctly.
    Pipeline is needed to save last indexes of train part """
    train_data, test_data = get_ts_data_with_dt_idx()
    # convert indexes to string
    train_data.idx = list(map(str, train_data.idx))
    test_data.idx = list(map(str, test_data.idx))
    train_pred_data = deepcopy(train_data)
    # old non int indexes
    old_train_data_idx = copy(train_data.idx)
    old_train_pred_data_idx = copy(train_pred_data.idx)
    old_test_data_idx = copy(test_data.idx)
    # pipeline object need to save last_fit_idx
    dummy_pipeline = Pipeline(PipelineNode("сut"))
    train_data = train_data.convert_non_int_indexes_for_fit(dummy_pipeline)
    train_pred_data = train_pred_data.convert_non_int_indexes_for_predict(dummy_pipeline)
    test_data = test_data.convert_non_int_indexes_for_predict(dummy_pipeline)
    # check is new integer indexes correct
    assert train_data.idx[-1] == train_pred_data.idx[0] - 1
    assert train_data.idx[-1] == test_data.idx[0] - 1
    # check is non integer indexes equal
    assert np.all(train_data.supplementary_data.non_int_idx == old_train_data_idx)
    assert np.all(train_pred_data.supplementary_data.non_int_idx == old_train_pred_data_idx)
    assert np.all(test_data.supplementary_data.non_int_idx == old_test_data_idx)


def test_data_convert_dt_indexes_correct():
    """ Test is datetime indexes converted correctly.
    Pipeline is needed to save last indexes of train part """
    train_data, test_data = get_ts_data_with_dt_idx()
    train_pred_data = deepcopy(train_data)
    # old non int indexes
    old_train_data_idx = copy(train_data.idx)
    old_train_pred_data_idx = copy(train_pred_data.idx)
    old_test_data_idx = copy(test_data.idx)
    # pipeline object need to save last_fit_idx
    dummy_pipeline = Pipeline(PipelineNode("сut"))
    train_data = train_data.convert_non_int_indexes_for_fit(dummy_pipeline)
    train_pred_data = train_pred_data.convert_non_int_indexes_for_predict(dummy_pipeline)
    test_data = test_data.convert_non_int_indexes_for_predict(dummy_pipeline)
    # check is new integer indexes correct (now we know order of indexes)
    assert train_data.idx[-1] == train_pred_data.idx[-1]
    assert train_data.idx[-1] == test_data.idx[0] - 1
    # check is non integer indexes equal
    assert np.all(train_data.supplementary_data.non_int_idx == old_train_data_idx)
    assert np.all(train_pred_data.supplementary_data.non_int_idx == old_train_pred_data_idx)
    assert np.all(test_data.supplementary_data.non_int_idx == old_test_data_idx)


@pytest.mark.parametrize('columns_to_use, possible_idx_keywords',
                         [
                             (None, ['b', 'c', 'a', 'some']),
                             (['b', 'c'], ['a', 'some'])
                         ])
def test_define_index_from_csv_with_first_index_column(columns_to_use, possible_idx_keywords):
    dummy_csv_path = fedot_project_root().joinpath('test/data/dummy.csv')
    df = get_df_from_csv(dummy_csv_path, delimiter=',',
                         columns_to_use=columns_to_use, possible_idx_keywords=possible_idx_keywords)
    assert df.index.name == 'a'
    assert np.array_equal(df.index, [1, 2, 3])
    assert np.array_equal(df.columns, ['b', 'c'])
    assert np.array_equal(df, list(zip([4, 5, 6], [7, 8, 9])))


def test_define_index_from_csv_with_non_first_index_column():
    dummy_csv_path = fedot_project_root().joinpath('test/data/dummy.csv')
    df = get_df_from_csv(dummy_csv_path, delimiter=',', columns_to_use=['b', 'c'],
                         possible_idx_keywords=['a', 'b', 'c', 'some'])
    assert df.index.name == 'b'
    assert np.array_equal(df.index, [4, 5, 6])
    assert np.array_equal(df.columns, ['c'])
    assert np.array_equal(df, [[7], [8], [9]])


def test_define_index_from_csv_without_index_column():
    dummy_csv_path = fedot_project_root().joinpath('test/data/dummy.csv')
    df = get_df_from_csv(dummy_csv_path, delimiter=',',
                         possible_idx_keywords=['some'])
    assert df.index.name is None
    assert np.array_equal(df.index, [0, 1, 2])
    assert np.array_equal(df.columns, ['a', 'b', 'c'])
    assert np.array_equal(df, list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9])))


def test_is_data_type_functions():
    data_types = [x for x in DataTypesEnum]
    array = np.array([0])
    for data_type in data_types:
        data = InputData(idx=array, features=array, target=array,
                         data_type=data_type, task=Task(TaskTypesEnum.clustering))
        attr_name = [key for key in dir(data) if key == f"is_{data_type.name}"][0]
        assert getattr(data, attr_name)
        for non_data_type in data_types:
            if non_data_type is not data_type:
                attr_name = [key for key in dir(data) if key == f"is_{non_data_type.name}"][0]
                assert not getattr(data, attr_name)


def test_is_task_type_functions():
    task_types = [x for x in TaskTypesEnum]
    array = np.array([0])
    for task_type in task_types:
        data = InputData(idx=array, features=array, target=array,
                         data_type=DataTypesEnum.table, task=Task(task_type))
        attr_name = [key for key in dir(data) if key == f"is_{task_type.name}"][0]
        assert getattr(data, attr_name)
        for non_task_type in task_types:
            if non_task_type is not task_type:
                attr_name = [key for key in dir(data) if key == f"is_{non_task_type.name}"][0]
                assert not getattr(data, attr_name)
