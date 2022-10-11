import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters

warnings.filterwarnings("ignore", category=UserWarning)


class DataSourceStrategy(EvaluationStrategy):

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return object()

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        return OutputData(idx=predict_data.idx, features=predict_data.features, task=predict_data.task,
                          data_type=predict_data.data_type, target=predict_data.target, predict=predict_data.features)

    def _convert_to_operation(self, operation_type: str):
        return object()
