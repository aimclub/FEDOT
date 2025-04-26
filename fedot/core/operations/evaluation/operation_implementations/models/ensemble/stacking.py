from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters

from sklearn.linear_model import (
    Ridge, LogisticRegression
)


class StackingImplementation(ModelImplementation):
    """Base class for stacking operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, input_data: InputData):
        """Fit the stacking model on input meta-data.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make predictions using the stacked model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        preds = self.model.predict(input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=preds)
        return output_data


class StackingClassifier(StackingImplementation):
    """Stacking implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LogisticRegression()  # ridge is base estimator

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Predict class probabilities using the stacked classifier.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        preds = self.model.predict_proba(input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=preds)
        return output_data


class StackingRegressor(StackingImplementation):
    """Stacking implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = Ridge()
