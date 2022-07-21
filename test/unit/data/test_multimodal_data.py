import os

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_multimodal_data_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_multimodal_classification.csv'
    task = Task(TaskTypesEnum.classification)
    df = pd.read_csv(os.path.join(test_file_path, file))
    text_data = np.array(df['description'])
    table_data = np.array(df.drop(columns=['id', 'description', 'variety']))
    target = np.array(df['variety'])
    idx = df['id']
    expected_text_features = InputData(features=text_data, target=target,
                                       idx=idx,
                                       task=task,
                                       data_type=DataTypesEnum.text).features
    expected_table_features = InputData(features=table_data, target=target,
                                        idx=idx,
                                        task=task,
                                        data_type=DataTypesEnum.table).features
    actual_data = MultiModalData.from_csv(os.path.join(test_file_path, file))
    actual_text_features = actual_data['data_source_text/description'].features
    actual_table_features = actual_data['data_source_table'].features
    assert np.array_equal(expected_text_features, actual_text_features)
    assert np.array_equal(expected_table_features, actual_table_features)


def test_multimodal_data_with_custom_target():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_multimodal_classification.csv'
    file_custom = '../../data/simple_multimodal_classification_with_custom_target.csv'

    file_data = MultiModalData.from_csv(os.path.join(test_file_path, file))

    expected_table_features = file_data['data_source_table'].features
    expected_target = file_data.target

    custom_file_data = MultiModalData.from_csv(os.path.join(test_file_path, file_custom))
    actual_table_features = custom_file_data['data_source_table'].features
    actual_target = custom_file_data.target

    assert not np.array_equal(expected_table_features, actual_table_features)
    assert not np.array_equal(expected_target, actual_target)

    custom_file_data = MultiModalData.from_csv(
        os.path.join(test_file_path, file_custom),
        columns_to_drop=['redundant'], target_columns='variety')

    actual_table_features = custom_file_data['data_source_table'].features
    actual_target = custom_file_data.target

    assert np.array_equal(expected_table_features, actual_table_features)
    assert np.array_equal(expected_target, actual_target)


def test_multi_modal_data():
    num_samples = 5
    target = np.asarray([0, 0, 1, 0, 1])
    img_data = InputData(idx=range(num_samples),
                         features=None,  # in test the real data is not passed
                         target=target,
                         data_type=DataTypesEnum.text,
                         task=Task(TaskTypesEnum.classification))
    tbl_data = InputData(idx=range(num_samples),
                         features=None,  # in test the real data is not passed
                         target=target,
                         data_type=DataTypesEnum.table,
                         task=Task(TaskTypesEnum.classification))

    multi_modal = MultiModalData({
        'data_source_img': img_data,
        'data_source_table': tbl_data,
    })

    assert multi_modal.task.task_type == TaskTypesEnum.classification
    assert len(multi_modal.idx) == 5
    assert multi_modal.num_classes == 2
    assert np.array_equal(multi_modal.target, target)

    # check setter
    new_target = np.asarray([1, 1, 1, 1, 1])
    multi_modal.target = new_target
    assert np.array_equal(multi_modal.target, new_target)
