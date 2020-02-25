from typing import List

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn import preprocessing


@dataclass
class Data:
    idx: np.array
    features: np.array

    @staticmethod
    def from_csv(file_path):
        data_frame = pd.read_csv(file_path)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        features = data_array[1:-1].T
        target = data_array[-1]
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


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
