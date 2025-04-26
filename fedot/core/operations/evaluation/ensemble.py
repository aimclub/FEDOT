from typing import Optional
from abc import abstractmethod
import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.stacking import (
    StackingClassifier, StackingRegressor)
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.voting import VotingClassifier
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.blending import (
    BlendingClassifier, BlendingRegressor)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.evaluation_interfaces import is_multi_output_task
from fedot.utilities.random import ImplementationRandomStateHandler
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class EnsembleStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the ensemble methods

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``blend_clf``-> BlendingClassifier
                - ``blend_reg``-> BlendingRegressor
                - ``stack_clf`` -> StackingClassifier
                - ``stack_reg`` -> StackingRegressor
                - ``voting`` -> VotingClassifier

        params: hyperparameters to fit the operation with
    """
    _operations_by_types = {
        'blend_clf': BlendingClassifier,
        'blend_reg': BlendingRegressor,
        'stack_clf': StackingClassifier,
        'stack_reg': StackingRegressor,
        'voting': VotingClassifier
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        if is_multi_output_task(train_data):
            raise ValueError(f'Ensemble methods do not support multi-output tasks.')

        operation_implementation = self.operation_impl(self.params_for_fit)

        if train_data.task.task_type == TaskTypesEnum.classification \
                and len(train_data.class_labels) == 2:  # if binary classification task
            expand_binary_input(train_data)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    @abstractmethod
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """This method used for prediction of the target data

        Args:
            trained_operation: operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        raise AbstractMethodNotImplementError


class EnsembleClassificationStrategy(EnsembleStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if len(predict_data.class_labels) == 2:
            expand_binary_input(predict_data)

        if self.output_mode == 'labels' or isinstance(trained_operation, VotingClassifier):
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode == 'default':
            prediction = trained_operation.predict_proba(predict_data)
        else:
            raise ValueError(f'Output mode {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class EnsembleRegressionStrategy(EnsembleStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)
        return self._convert_to_output(prediction, predict_data)


def expand_binary_input(train_data: InputData) -> None:
    """Expands binary classification probabilities array by adding complementary probabilities.

    Example:
        Before transformation (3 models): [P1_m1, P1_m2, P1_m3]
        After transformation: [P0_m1, P1_m1, P0_m2, P1_m2, P0_m3, P1_m3]

    Args:
        train_data: InputData with P(class=1) probabilities for each model

    Returns:
        None: Modifies train_data.features in-place
    """
    full_probs_arr = np.zeros((train_data.features.shape[0], train_data.features.shape[1] * 2))
    full_probs_arr[:, 1::2] = train_data.features  # even columns (0, 2, 4...) - P(class=1)
    full_probs_arr[:, 0::2] = 1 - train_data.features  # odd columns (1, 3, 5...) - P(class=0)
    train_data.features = full_probs_arr
