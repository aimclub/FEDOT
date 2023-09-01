import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def get_multimodal_data(input_data_count=5, length=100, feature_count=5):
    data = MultiModalData()
    for i in range(input_data_count):
        data[str(i)] = InputData(idx=np.arange(length),
                                 features=np.random.rand(length, feature_count),
                                 target=np.random.rand(length),
                                 data_type=DataTypesEnum.table,
                                 task=Task(TaskTypesEnum.regression))
    return data


def test_multi_modal_data():
    """
    Checking basic functionality of MultiModalData class.
    """
    num_samples = 5
    target = np.asarray([0, 0, 1, 0, 1])
    image_data = InputData(idx=range(num_samples),
                           features=None,  # in test the real data is not passed
                           target=target,
                           data_type=DataTypesEnum.text,
                           task=Task(TaskTypesEnum.classification))
    table_data = InputData(idx=range(num_samples),
                           features=None,  # in test the real data is not passed
                           target=target,
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))

    multi_modal = MultiModalData({
        'data_source_img': image_data,
        'data_source_table': table_data,
    })

    assert multi_modal.task.task_type == TaskTypesEnum.classification
    assert len(multi_modal.idx) == 5
    assert np.array_equal(multi_modal.idx, range(num_samples))
    assert multi_modal.num_classes == 2
    assert np.array_equal(multi_modal.class_labels, [0, 1])
    assert np.array_equal(multi_modal.target, target)
    assert np.array_equal(multi_modal.data_type, [image_data.data_type, table_data.data_type])

    # check setter
    new_target = np.asarray([1, 1, 1, 1, 1])
    multi_modal.target = new_target
    for input_data in multi_modal.values():
        assert np.array_equal(input_data.target, new_target)
    assert np.array_equal(multi_modal.target, new_target)

    new_idx = np.asarray([1, 1, 1, 1, 1])
    multi_modal.idx = new_idx
    for input_data in multi_modal.values():
        assert np.array_equal(input_data.idx, new_idx)
    assert np.array_equal(multi_modal.idx, new_idx)

    new_task = Task(TaskTypesEnum.regression)
    multi_modal.task = new_task
    for input_data in multi_modal.values():
        assert input_data.task == new_task
    assert multi_modal.task == new_task


def test_multimodal_data_from_csv():
    """
    Checking correctness of MultiModalData import from csv file.
    """
    file_path = 'test/data/simple_multimodal_classification.csv'
    path = Path(fedot_project_root(), file_path)
    df = pd.read_csv(path)
    text_data = np.array(df['description'])
    table_data = np.array(df.drop(columns=['id', 'description', 'variety']))
    target = np.array(df['variety']).reshape(-1, 1)
    actual_data = MultiModalData.from_csv(path)
    actual_text_features = actual_data['data_source_text/description'].features
    actual_table_features = actual_data['data_source_table'].features
    actual_target = actual_data.target

    assert np.array_equal(actual_text_features, text_data)
    assert np.array_equal(actual_table_features, table_data)
    assert np.array_equal(actual_target, target)


def test_multimodal_data_with_custom_target():
    """
    Checking that MultiModalData imports last column as target by default
    and that manual set of target and redundant columns works as expected.
    """
    file_path = 'test/data/simple_multimodal_classification.csv'
    path = Path(fedot_project_root(), file_path)
    file_data = MultiModalData.from_csv(path)

    expected_table_features = file_data['data_source_table'].features
    expected_target = file_data.target

    file_custom_path = 'test/data/simple_multimodal_classification_with_custom_target.csv'
    path_custom = Path(fedot_project_root(), file_custom_path)
    custom_file_data = MultiModalData.from_csv(path_custom)
    actual_table_features = custom_file_data['data_source_table'].features
    actual_target = custom_file_data.target

    assert not np.array_equal(expected_table_features, actual_table_features)
    assert not np.array_equal(expected_target, actual_target)

    custom_file_data = MultiModalData.from_csv(path_custom,
                                               columns_to_drop=['redundant'], target_columns='variety')

    actual_table_features = custom_file_data['data_source_table'].features
    actual_target = custom_file_data.target

    assert np.array_equal(expected_table_features, actual_table_features)
    assert np.array_equal(expected_target, actual_target)


@pytest.mark.parametrize('data_type', [DataTypesEnum.text, DataTypesEnum.table])
def test_text_data_only(data_type):
    """
    Checking cases when there is only one data source.
    """
    if data_type is DataTypesEnum.text:
        # Case when there is no table data in csv, but MultiModalData.from_csv() is used
        file_path = 'test/data/simple_multimodal_classification_text.csv'
        data_source_name = 'data_source_text/description'
    elif data_type is DataTypesEnum.table:
        # Case when there is no text data in csv, but MultiModalData.from_csv() is used
        file_path = 'test/data/simple_classification.csv'
        data_source_name = 'data_source_table'

    path = Path(fedot_project_root(), file_path)
    file_data = InputData.from_csv(path, data_type=DataTypesEnum.text)
    file_mm_data = MultiModalData.from_csv(path)

    assert len(file_mm_data) == 1
    assert file_mm_data[data_source_name].data_type is data_type
    assert file_mm_data[data_source_name].features.all() == file_data.features.all()
    assert file_mm_data[data_source_name].target.all() == file_data.target.all()


def test_multimodal_data_with_complicated_types():
    """
    Combines complicated table data with some text columns.
    For more detailed description of the table part
    of dataset look at data_with_complicated_types.
    """
    file_path = 'test/data/multimodal_data_with_complicated_types.csv'
    path = Path(fedot_project_root(), file_path)
    file_mm_data = MultiModalData.from_csv(path)
    model = Fedot(problem='classification')
    model.fit(features=file_mm_data,
              target=file_mm_data.target,
              predefined_model='auto')
    model.predict(file_mm_data)

    assert len(file_mm_data) == 2
    assert 'data_source_text/5' in file_mm_data
    assert file_mm_data['data_source_table'].features.shape == (18, 11)


@pytest.mark.parametrize(['start', 'stop', 'step'],
                         [(None, 10, None),
                          (2, 20, 3),
                          (None, -20, -2)])
def test_multimodal_slice(start, stop, step):
    data = get_multimodal_data()
    sliced = data.slice(start, stop, step)
    for indata1, indata2 in zip(data.values(), sliced.values()):
        indata1 = indata1.slice(start, stop, step)
        assert np.array_equal(indata1.idx, indata2.idx)
        assert np.array_equal(indata1.features, indata2.features)
        assert np.array_equal(indata1.target, indata2.target)


@pytest.mark.parametrize(['indexes', 'step'],
                         [([0, 1, 2], 1),
                          ([-3, -2, -1], 1),
                          ([-1, -2, -3], -1),
                          ([0, 3, 6], 3)
                          ])
def test_slice_by_index(indexes, step):
    data = get_multimodal_data()
    sliced = data.slice_by_index(indexes, step)
    for indata1, indata2 in zip(data.values(), sliced.values()):
        indata1 = indata1.slice_by_index(indexes, step)
        assert np.array_equal(indata1.idx, indata2.idx)
        assert np.array_equal(indata1.features, indata2.features)
        assert np.array_equal(indata1.target, indata2.target)
