from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum


@dataclass
class Data:
    idx: np.array
    features: np.array
    task: Task
    data_type: DataTypesEnum

    @staticmethod
    def from_csv(file_path, delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 with_target=True):
        data_frame = pd.read_csv(file_path, sep=delimiter)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        if with_target:
            features = data_array[1:-1].T
            target = data_array[-1].astype(np.float)
        else:
            features = data_array[1:].T
            target = None
        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)


@dataclass
class InputData(Data):
    target: np.array = None

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    @staticmethod
    def from_predictions(outputs: List['OutputData'], target: np.array):
        if len(set([output.data_type for output in outputs])) > 1:
            raise ValueError('Inconsistent data types')
        if len(set([output.task.task_type for output in outputs])) > 1:
            raise ValueError('Inconsistent task types')

        task = outputs[0].task
        data_type = outputs[0].data_type
        idx = outputs[0].idx

        dataset_merging_funcs = {
            DataTypesEnum.ts: _combine_datasets_ts,
            DataTypesEnum.table: _combine_datasets_table
        }
        dataset_merging_funcs.setdefault(data_type, _combine_datasets_common)

        features = dataset_merging_funcs[data_type](outputs)

        return InputData(idx=idx, features=features, target=target, task=task,
                         data_type=data_type)


@dataclass
class OutputData(Data):
    predict: np.array = None


def split_train_test(data, split_ratio=0.8, with_shuffle=False):
    assert 0. <= split_ratio <= 1.
    if with_shuffle:
        data_train, data_test = train_test_split(data, test_size=1. - split_ratio, random_state=42)
    else:
        split_point = int(len(data) * split_ratio)
        data_train, data_test = data[:split_point], data[split_point:]
    return data_train, data_test


def _convert_dtypes(data_frame: pd.DataFrame):
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def train_test_data_setup(data: InputData, split_ratio=0.8, shuffle_flag=False) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features, split_ratio, with_shuffle=shuffle_flag)
    train_data_y, test_data_y = split_train_test(data.target, split_ratio, with_shuffle=shuffle_flag)
    train_idx, test_idx = split_train_test(data.idx, split_ratio)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx, task=data.task, data_type=data.data_type)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx, task=data.task,
                          data_type=data.data_type)
    return train_data, test_data


def _combine_datasets_ts(outputs: List[OutputData]):
    features = list()
    expected_len = max([len(output.idx) for output in outputs])
    for elem in outputs:
        predict = elem.predict
        if len(elem.predict) != expected_len:
            if isinstance(elem.predict, list):
                predict = np.zeros(expected_len - len(elem.predict)) + elem.predict
            else:
                predict = np.concatenate((np.zeros(expected_len - len(elem.predict)),
                                          elem.predict))

        features.append(predict)

    features = np.array(features).T

    return features


def _combine_datasets_table(outputs: List[OutputData]):
    features = list()
    expected_len = len(outputs[0].predict)
    for elem in outputs:
        if len(elem.predict) != expected_len:
            raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])

    features = np.array(features).T

    return features


def _combine_datasets_common(outputs: List[OutputData]):
    features = list()

    for elem in outputs:
        if len(elem.predict) != len(outputs[0].predict):
            raise NotImplementedError(f'Non-equal prediction length: '
                                      f'{len(elem.predict)} and {len(outputs[0].predict)}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])
    return features
