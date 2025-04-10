from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class BaggingImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, input_data: InputData):
        pass

    def predict(self, input_data: InputData) -> OutputData:
        pass


class BaggingClassifierImplementation(BaggingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = BaggingClassifier(self.params.to_dict().items())


class BaggingRegressorImplementation(BaggingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = BaggingRegressor(self.params.to_dict().items())
