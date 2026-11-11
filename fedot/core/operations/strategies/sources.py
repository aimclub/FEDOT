import warnings
from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters

warnings.filterwarnings("ignore", category=UserWarning)


class DataSourceStrategy(EvaluationStrategy):

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        return object()

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        return self._replace_features_in_tensor_data(predict_data.features, predict_data)

    def _convert_to_operation(self, operation_type: str):
        return object()
