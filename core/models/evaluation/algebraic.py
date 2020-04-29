from typing import List

import numpy as np


def plus_operator(inputs: List[float]):
    return np.add(inputs[:, 0], inputs[:, 1])


def minus_operator(inputs: List[float]):
    return np.subtract(inputs[:, 0], inputs[:, 1])


def mult_operator(inputs: List[float]):
    return np.multiply(inputs[:, 0], inputs[:, 1])


def div_operator(inputs: List[float]):
    # return np.divide(inputs[:, 0], inputs[:, 1], where=inputs[:, 1] != 0)
    return inputs[:, 0]
