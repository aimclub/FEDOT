from typing import Optional

from perpetual import PerpetualBooster

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


class PerpetualImplementation(ModelImplementation):
    """Base class for perpetual gbm"""
    __operation_params = ['n_jobs']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.seed = 42
        self.model = None
        self.classes_ = None

    def fit(self, input_data: InputData):
        """Fit the perpetual gbm.
        Args:
            input_data: Input data features.
        """
        self.classes_ = input_data.class_labels
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make labels predictions using the perpetual gbm.
        Args:
            input_data: Input data features.
        """
        labels = self.model.predict(X=input_data.features)
        # output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return labels

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Make probabilities predictions using the perpetual gbm.
        Args:
            input_data: Input data features.
        """
        if input_data.task == TaskTypesEnum.regression or input_data.task == TaskTypesEnum.ts_forecasting:
            raise ValueError('This method does not support regression or time series forecasting tasks')

        probs = self.model.predict_proba(X=input_data.features)
        # output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return probs


class PerpetualClassifier(PerpetualImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = PerpetualBooster(objective='LogLoss')


class PerpetualRegressor(PerpetualImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = PerpetualBooster(objective='SquaredLoss')
