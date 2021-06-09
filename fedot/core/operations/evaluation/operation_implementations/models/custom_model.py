from typing import Optional
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation

class CustomModelImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.input_data=None

    def fit(self, input_data):
        self.input_data = input_data
        pass

    def get_params(self):
        return self.params

    def predict(self):
        return self.input_data


