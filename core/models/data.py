from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing


@dataclass
class Data:
    idx: np.array
    features: np.array

    @staticmethod
    def from_csv(file_path, delimiter=',', normalization=False, label=[]):

        data_frame = pd.read_csv(file_path, sep=delimiter)
        idx = np.array(data_frame[data_frame.columns[0]].T)
        if not label:
            label = data_frame.columns[len(data_frame.columns) - 1]

        target = np.array(data_frame[label].T).astype(np.float)
        data_frame = data_frame.drop([label] + [data_frame.columns[0]], axis=1)
        if normalization:
            data_frame = normalize_data(data_frame)
        features = np.array(data_frame)

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


def normalize_data(train):
    for column in train.columns:
        col = train[[column]].values.astype(float)
        if not col.min() in range(0, 1) or not col.max() in range(0, 1):
            col = preprocessing.MinMaxScaler().fit_transform(col)
            train[[column]] = col
    return train


def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
