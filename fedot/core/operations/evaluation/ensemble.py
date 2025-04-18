from typing import Optional

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
            raise ValueError(f'Ensemble methods not support multi-output task')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)

        return self._convert_to_output(prediction, predict_data)
