from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Data:
    idx: np.array
    features: np.array

    @staticmethod
    def from_csv(file_path, delimiter=','):
        data_frame = pd.read_csv(file_path, sep=delimiter)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        features = data_array[1:-1].T
        target = data_array[-1].astype(np.float)
        return InputData(idx=idx, features=features, target=target)

    @staticmethod
    def from_predictions(outputs: List['OutputData'], target: np.array):
        idx = outputs[0].idx
        features = list()
        for elem in outputs:
            features.append(elem.predict)
        return InputData(idx=idx, features=np.array(features).T, target=target)


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


def train_test_data_setup(data: InputData) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features)
    train_data_y, test_data_y = split_train_test(data.target)
    train_idx, test_idx = split_train_test(data.idx)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx)
    return train_data, test_data
