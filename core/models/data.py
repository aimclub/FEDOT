from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing


@dataclass
class Data:
    idx: np.array
    features: np.array
    target: np.array
    prediction: Optional[np.array] = None

    @staticmethod
    def from_csv(file_path):
        data_frame = pd.read_csv(file_path)
        data_array = np.array(data_frame).T
        idx = data_array[0]
        features = data_array[1:-1].T
        target = data_array[-1]
        return Data(idx=idx, features=features, target=target)

    @staticmethod
    def from_vectors(vectors: List[np.array]):
        features = np.array(vectors[:-1]).T
        idx = np.arange(0, vectors[0].size)
        target = vectors[-1]
        return Data(idx=idx, features=features, target=target)


def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
