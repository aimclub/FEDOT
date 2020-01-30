import numpy as np


class DataStream:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
