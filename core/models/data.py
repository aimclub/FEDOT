from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum


@dataclass
class Data:
    idx: np.array
    features: np.array
    task_type: TaskTypesEnum

    @staticmethod
    def from_csv(file_path, delimiter=',', target_flag=False,
                 task_type: TaskTypesEnum = MachineLearningTasksEnum.classification):
        data_frame = pd.read_csv(file_path, sep=delimiter)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        if target_flag:
            features = data_array[1:].T
            target = None
        else:
            features = data_array[1:-1].T
            target = data_array[-1].astype(np.float)
        return InputData(idx=idx, features=features, target=target, task_type=task_type)

    @staticmethod
    def from_predictions(outputs: List['OutputData'], target: np.array):
        task_type = outputs[0].task_type
        idx = outputs[0].idx
        features = list()
        expected_len = len(outputs[0].predict)
        for elem in outputs:
            if len(elem.predict) != expected_len:
                raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
            features.append(elem.predict)

        if task_type == MachineLearningTasksEnum.classification and len(outputs) >= 2:
            for prediction in features:
                if len(prediction.shape) >= 2:
                    feat_final = np.concatenate(features, axis=1)
                else:
                    feat_final = np.concatenate([x.reshape(-1, 1) for x in features], axis=1)
        else:
            feat_final = np.array(features).T

        if len(feat_final.shape) >= 3:
            num_of_samples = feat_final.shape[1]
            feat_final = feat_final.reshape(num_of_samples, -1)

        return InputData(idx=idx, features=feat_final, target=target, task_type=task_type)


@dataclass
class InputData(Data):
    target: np.array

    @property
    def num_classes(self):
        if self.task_type == MachineLearningTasksEnum.classification:
            return len(np.unique(self.target))
        else:
            return None


@dataclass
class OutputData(Data):
    predict: np.array


def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def _convert_dtypes(data_frame: pd.DataFrame):
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def train_test_data_setup(data: InputData, split_ratio=0.8) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features, split_ratio)
    train_data_y, test_data_y = split_train_test(data.target, split_ratio)
    train_idx, test_idx = split_train_test(data.idx, split_ratio)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx, task_type=data.task_type)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx, task_type=data.task_type)
    return train_data, test_data
