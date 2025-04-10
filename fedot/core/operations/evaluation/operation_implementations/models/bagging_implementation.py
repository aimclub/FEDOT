from typing import Optional

from sklearn.ensemble import BaggingRegressor, BaggingClassifier

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class BaggingImplementation(ModelImplementation):
    """ Bagging ensemble implementation from scikit-learn """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, input_data: InputData):
        """
        Build a Bagging ensemble of estimators from the training set
        (Decision Tree estimator by default)
        """
        self.model.fit(X=input_data.features, y=input_data.target)

    def predict(self, input_data: InputData) -> OutputData:
        return self.model.predict(input_data.features)


class BaggingClassifierImplementation(BaggingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = BaggingClassifier(self.params.to_dict().items())


class BaggingRegressorImplementation(BaggingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = BaggingRegressor(self.params.to_dict().items())
