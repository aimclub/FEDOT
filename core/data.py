from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataStream:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y


@dataclass
class Data:
    features: np.array
    target: np.array
    idx: np.array

    @staticmethod
    def from_csv(file_path):
        data_frame = pd.read_csv(file_path)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        features = data_array[1:-1].T
        target = data_array[-1]
        return Data(features, target, idx)


def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
