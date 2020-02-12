from dataclasses import dataclass

import numpy as np
import pandas as pd


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
