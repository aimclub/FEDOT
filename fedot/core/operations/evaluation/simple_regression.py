from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.torch import (
    TorchLinearRegressor,
    TorchMLPRegressor,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


class SimpleRegressionStrategy(EvaluationStrategy):
    """Lightweight regression models trained natively on TensorData tensors."""

    _operations_by_types = {
        'torch_linear_reg': TorchLinearRegressor,
        'torch_mlp_reg': TorchMLPRegressor,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        if predict_data.task.task_type is not TaskTypesEnum.regression:
            raise ValueError('Torch regression models support only regression task')
        prediction = trained_operation.predict(predict_data)
        return self._replace_predict_in_tensor_data(prediction, predict_data)
