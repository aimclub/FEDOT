from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


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
        target = data_array[-1].astype(np.float)
        return InputData(idx=idx, features=features, target=target)

    @staticmethod
    def from_csv_special(file_path):
        data_frame = pd.read_csv(file_path)
        data_array = np.array(data_frame).T
        idx = list(range(len(data_frame)))
        data_array = data_array[np.r_[0:4, 6:13]]
        features = data_array[1:].T
        target = data_array[0].astype(np.float)
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


def preprocess(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    return preprocessing.scale(x)
