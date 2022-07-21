import os
from copy import deepcopy, copy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from test.unit.tasks.test_classification import get_image_classification_data
from test.unit.tasks.test_forecasting import get_ts_data_with_dt_idx


@pytest.fixture()
def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def test_data_subset_correct(data_setup):
    subset_size = 50
    subset = data_setup.subset_range(0, subset_size - 1)

    assert len(subset.idx) == subset_size
    assert len(subset.features) == subset_size
    assert len(subset.target) == subset_size


def test_data_subset_incorrect(data_setup):
    subset_size = 105
    with pytest.raises(ValueError):
        assert data_setup.subset_range(0, subset_size)

    with pytest.raises(ValueError):
        assert data_setup.subset_range(-1, subset_size)
    with pytest.raises(ValueError):
        assert data_setup.subset_range(-1, -1)


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


def test_data_from_image():
    _, _, dataset_to_validate = get_image_classification_data()

    assert dataset_to_validate.data_type == DataTypesEnum.image
    assert type(dataset_to_validate.features) == np.ndarray
    assert type(dataset_to_validate.target) == np.ndarray


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

    assert not np.array_equal(data.idx, shuffled_data.idx)
    assert not np.array_equal(data.features, shuffled_data.features)
    assert not np.array_equal(data.target, shuffled_data.target)

    assert np.array_equal(sorted(data.idx), sorted(shuffled_data.idx))


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
    dummy_pipeline = Pipeline(PrimaryNode("сut"))
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
    dummy_pipeline = Pipeline(PrimaryNode("сut"))
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
