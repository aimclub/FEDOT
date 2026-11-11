from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.models.neural import TorchLinearClassifier
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.schemas import validate_classification_output_mode
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler


class NeuralTabularStrategy(EvaluationStrategy):
    """Shared TensorData contract for tabular neural model implementations."""

    _operations_by_types = {}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation


class NeuralTabularClassificationStrategy(NeuralTabularStrategy):
    _operations_by_types = {
        'torch_linear': TorchLinearClassifier,
    }

    def predict(self, trained_operation, predict_data: TensorData):
        output_mode = validate_classification_output_mode(self.output_mode)
        if output_mode == 'labels':
            return trained_operation.predict(predict_data)
        prediction = trained_operation.predict_proba(predict_data)
        proba = prediction.predict
        n_classes = proba.shape[1]
        if n_classes < 2:
            raise ValueError('Data set contains only 1 target class. Please reformat your data.')
        if n_classes == 2 and output_mode != 'full_probs':
            proba = proba[:, 1]
        return self._replace_predict_in_tensor_data(proba, prediction)


class NeuralTabularRegressionStrategy(NeuralTabularStrategy):
    _operations_by_types = {
    }

    def predict(self, trained_operation, predict_data: TensorData):
        if predict_data.task.task_type is not TaskTypesEnum.regression:
            raise ValueError(
                'Neural tabular regression models support only regression task')
        return trained_operation.predict(predict_data)
