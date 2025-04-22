from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.blending import (
    BlendingClassifier, BlendingRegressor)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.evaluation_interfaces import is_multi_output_task
from fedot.utilities.random import ImplementationRandomStateHandler


class EnsembleStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the ensemble methods

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``blend_clf``-> BlendingClassifier
                - ``blend_reg``-> BlendingRegressor

        params: hyperparameters to fit the operation with
    """
    _operations_by_types = {
        'blend_clf': BlendingClassifier,
        'blend_reg': BlendingRegressor,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        if is_multi_output_task(train_data):
            raise ValueError(f'Ensemble methods do not support multi-output tasks.')

        operation_implementation = self.operation_impl(self.params_for_fit)

        if len(train_data.class_labels) == 2:  # if binary classification task
            expand_binary_input(train_data)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


def expand_binary_input(train_data: InputData) -> None:
    """Expands binary classification probabilities array by adding complementary probabilities.

    Example:
        Before transformation (3 models): [P0_m1, P0_m2, P0_m3]
        After transformation: [P0_m1, P1_m1, P0_m2, P1_m2, P0_m3, P1_m3]

    Args:
        train_data: InputData with P(class=0) probabilities for each model

    Returns:
        None: Modifies train_data.features in-place
    """
    full_probs_arr = np.zeros((train_data.features.shape[0], train_data.features.shape[1] * 2))
    full_probs_arr[:, 0::2] = train_data.features  # Even columns (0, 2, 4...) - P(class=0)
    full_probs_arr[:, 1::2] = 1 - train_data.features  # Odd columns (1, 3, 5...) - P(class=1)
    train_data.features = full_probs_arr
