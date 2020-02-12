from dataclasses import dataclass

import numpy as np
from sklearn import preprocessing


class DataStream:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y


@dataclass
class Data:
    features: np.array
    target: np.array

    @staticmethod
    def from_csv(self, file_path):
        raise NotImplementedError()


def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
