from abc import abstractmethod
from typing import List

import numpy as np


class Operator:
    def __init__(self, variables_ids):
        self.variables_ids = variables_ids

    @abstractmethod
    def evaluate(self, inputs: List[float]):
        raise NotImplementedError()


class PlusOperator(Operator):
    def evaluate(self, inputs: List[float]):
        return np.add(inputs[:, self.variables_ids[0]], inputs[:, self.variables_ids[1]])


class MinusOperator(Operator):
    def evaluate(self, inputs: List[float]):
        return np.subtract(inputs[:, self.variables_ids[0]], inputs[:, self.variables_ids[1]])


class MultOperator(Operator):
    def evaluate(self, inputs: List[float]):
        return np.multiply(inputs[:, self.variables_ids[0]], inputs[:, self.variables_ids[1]])


class DivOperator(Operator):
    def evaluate(self, inputs: List[float]):
        # return np.divide(inputs[:, self.variables_ids[0]], inputs[:, 1], where=inputs[:, 1] != 0)
        return inputs[:, 0]
