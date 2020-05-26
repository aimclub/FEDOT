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
    def from_csv(file_path, delimiter=',',
                 task_type: TaskTypesEnum = MachineLearningTasksEnum.classification):
        if task_type == MachineLearningTasksEnum.forecasting:
            raise ValueError('For forecasting, please use Data.from_npy method')
        data_frame = pd.read_csv(file_path, sep=delimiter)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        features = data_array[1:-1].T
        target = data_array[-1].astype(np.float)
        return InputData(idx=idx, features=features, target=target, task_type=task_type)

    @staticmethod
    def from_npy(filepath_features, filepath_target, idx, prediction_len=1):
        """
        Use this function only for forecasting
        filepath_features: str
            Features file with 3d input array - (n, window_len, features_dim).

        filepath_target: str
            Target file with 3d input array - (n, window_len, target_dim).
            After prediction for use only forecasting data get from last `prediction_len` timestamps.
    
        idx: str or np.array
            Index of train/target data
            If str, then tries to load from file with that path. Else uses np.array

        prediction_len: int
            Number of timestamps used as target
        """
        if prediction_len != 1:
            raise NotImplementedError('For now only 1 value forecasting is supported')

        features = np.load(filepath_features)
        target = np.load(filepath_target)
        assert features.ndim == target.ndim == 3, 'Features and target must be 3d datasets'
        assert features.shape[:2] == target.shape[:2], 'First two dimensions of features and target must be equal'
 
        if isinstance(idx, str):
            idx = np.load(idx)

        return InputData(idx=idx, features=features, target=target, task_type=MachineLearningTasksEnum.forecasting)

    @staticmethod
    def from_predictions(outputs: List['OutputData'], target: np.array):
        task_type = outputs[0].task_type
        idx = outputs[0].idx

        if task_type == MachineLearningTasksEnum.forecasting:
            features = np.concatenate([output.predict for output in outputs], axis=-1)
            return InputData(idx=idx, features=features, target=target, task_type=task_type)

        features = list()
        expected_len = len(outputs[0].predict)
        for elem in outputs:
            if len(elem.predict) != expected_len:
                raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
            features.append(elem.predict)
        return InputData(idx=idx, features=np.array(features).T, target=target, task_type=task_type)


@dataclass
class InputData(Data):
    target: np.array


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
