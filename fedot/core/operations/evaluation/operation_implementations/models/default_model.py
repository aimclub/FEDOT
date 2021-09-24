import numpy as np
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from typing import Callable
from typing import Optional

from fedot.core.log import Log

class DefaultImplementation(ModelImplementation):
    def __init__(self, model_info: str, model_alg: Callable[[dict, np.array], np.array], log: Log = None,  **params):
        super().__init__(log)
        self.model_info = model_info
        self.params = params
        self.model = model_alg

    def fit(self, input_data):
        """ Fitting of custom models are combined with prediction by user"""
        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        pass