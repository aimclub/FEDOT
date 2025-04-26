from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MetaDTImplementation(ModelImplementation):
    """Base class for meta decision tree ensemble implementation"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, input_data: InputData):
        """Fit the meta decision tree model on input meta-data.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make predictions using the stacked meta decision tree model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        preds = self.model.predict(input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=preds)
        return output_data


class MetaDTClassifier(MetaDTImplementation):
    """Meta decision tree implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = DecisionTreeClassifier(max_depth=3)

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Predict class probabilities using the stacked classifier.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        preds = self.model.predict_proba(input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=preds)
        return output_data


class MetaDTRegressor(MetaDTImplementation):
    """Meta decision tree implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = DecisionTreeRegressor(max_depth=3)
