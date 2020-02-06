from dataclasses import dataclass

import numpy as np


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
