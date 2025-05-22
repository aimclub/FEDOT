from abc import abstractmethod
from typing import Optional

from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse
from sklearn.linear_model import Ridge, Lasso, Lars, LogisticRegression

from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data_split import train_test_data_setup
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class StackingImplementation(ModelImplementation):
    """Base class for stacking operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.log = default_log('Stacking')
        self.model = None
        self.metric = None

    @abstractmethod
    def fit(self, input_data: InputData):
        raise AbstractMethodNotImplementError

    def predict(self, input_data: InputData) -> OutputData:
        """Make labels predictions using the stacked model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        labels = self.model.predict(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data


class StackingClassifier(StackingImplementation):
    """Stacking implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy

    def fit(self, input_data: InputData):
        """Fit the stacking model on input meta-data.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        train, val = train_test_data_setup(input_data, split_ratio=0.9)
        penalties = ['l2', 'l1', None]
        model_score_array = []

        for pen in penalties:
            model = LogisticRegression(penalty=pen, solver='saga')
            model.fit(train.features, train.target)
            predictions = model.predict(val.features)
            score = round(self.metric(val.target, predictions), 3)
            model_score_array.append((model, score))

        best_model, best_score = max(model_score_array, key=lambda x: x[1])
        self.model = best_model

        self.log.message(f"Stacking models with their score {model_score_array}. "
                         f"Chosen {best_model} with best validation score {best_score}")

        return self

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Make probabilities predictions using the stacked model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        probs = self.model.predict_proba(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data


class StackingRegressor(StackingImplementation):
    """Stacking implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = mse

    def fit(self, input_data: InputData):
        """Fit the stacking model on input meta-data.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        train, val = train_test_data_setup(input_data, split_ratio=0.9)
        models = [Ridge, Lasso, Lars]
        model_score_array = []

        for model in models:
            model = model()
            model.fit(train.features, train.target)
            predictions = model.predict(val.features)
            score = round(self.metric(val.target, predictions), 3)
            model_score_array.append((model, score))

        best_model, best_score = max(model_score_array, key=lambda x: x[1])
        self.model = best_model

        self.log.message(f"Stacking models with their score {model_score_array}. "
                         f"Chosen {best_model} with best validation score {best_score}")

        return self
