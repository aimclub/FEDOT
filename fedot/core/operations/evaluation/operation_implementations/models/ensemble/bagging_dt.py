from typing import Optional

from golem.core.log import default_log
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class BaggingImplementation(ModelImplementation):
    """Base class for bagging operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.log = default_log('Bagging')

        self.seed = 42
        self.model = None

    def fit(self, input_data: InputData):
        """Fit the bagging model. Decision Tree estimator set as default.

        Args:
            input_data: Input data features.
        """
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make labels predictions using the bagging model.

        Args:
            input_data: Input data features.
        """
        labels = self.model.predict(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data


class BaggingClassificationImplementation(BaggingImplementation):
    """Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  DecisionTreeClassifier(max_depth=3)
        self.model = BaggingClassifier(estimator=est)

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Make probabilities predictions using the bagging model.

        Args:
            input_data: Input data features.
        """
        probs = self.model.predict_proba(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data


class BaggingRegressionImplementation(BaggingImplementation):
    """Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  DecisionTreeRegressor(max_depth=3)
        self.model = BaggingRegressor(estimator=est)
