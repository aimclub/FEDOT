import warnings
from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.features_reducing import (
    PCAImplementation,
    TruncatedSVDImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings('ignore', category=UserWarning)


class TensorTransformStrategy(EvaluationStrategy):
    """Dispatcher for TensorData transform operations (FE / dimensionality ops).

    Analogous to :class:`FedotPreprocessingStrategy`, but for ``TensorData``
    end-to-end nodes (not multi-step cleaning like optional preprocessing).

    Supported operations:
        - ``pca`` → :class:`PCAImplementation`
        - ``truncated_svd`` → :class:`TruncatedSVDImplementation`

    Invariant:
        ``predict_for_fit`` is intentionally identical to ``predict`` for current
        transforms (same projection on fit and predict batches).
        Override ``predict_for_fit`` only if a later op needs a different fit-stage
        path (e.g. leakage-prone behaviour).
    """

    _operations_by_types = {
        'pca': PCAImplementation,
        'truncated_svd': TruncatedSVDImplementation,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        return trained_operation.transform(predict_data)

    def predict_for_fit(self, trained_operation, predict_data: TensorData) -> TensorData:
        return self.predict(trained_operation, predict_data)
