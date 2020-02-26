from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing


@dataclass
class Data:
    idx: np.array
    features: np.array
    target: np.array

    @staticmethod
    def from_csv(file_path, delimiter=',', normalization=False, label=[]):

        data_frame = pd.read_csv(file_path, sep=delimiter)
        idx = np.array(data_frame[data_frame.columns[0]].T)
        if not label:
            label = data_frame.columns[len(data_frame.columns)-1]

        target = np.array(data_frame[label].T)
        data_frame = data_frame.drop([label] + [data_frame.columns[0]], axis=1)


        if normalization:
            data_frame = normalize(data_frame)
        print(data_frame.head())
        features = np.array(data_frame)
        # features = data_array[1:-1].T

        print("features", features)
        print("idx", idx)
        print("target", target)

        return Data(idx=idx, features=features, target=target)

    @staticmethod
    def from_vectors(vectors: List[np.array]):
        features = np.array(vectors[:-1]).T
        idx = np.arange(0, vectors[0].size)
        target = vectors[-1]
        return Data(idx=idx, features=features, target=target)


def normalize(train):
    for column in train.columns:
        col = train[[column]].values.astype(float)
        if not col.min() in range(0, 1) or not col.max() in range(0, 1):
            col = preprocessing.MinMaxScaler().fit_transform(col)
            train[[column]] = col
    return train

def split_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

'''
def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x, axis=1)
'''
