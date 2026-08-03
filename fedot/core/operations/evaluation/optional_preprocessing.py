from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.preprocessing.service.optional_service import OptionalService
from fedot.preprocessing.service.tensor_optional_runtime import (
    get_optional_runtime_spec_for_tensor_data,
)


class TensorOptionalPreprocessingStrategy(EvaluationStrategy):
    """TensorData strategy that fits and applies optional preprocessing services.

    Unlike :class:`FedotPreprocessingStrategy`, this strategy works with
    ``TensorData`` end-to-end and stores a fitted :class:`OptionalService`
    as the node operation. Runtime service selection (tabular/TS) happens in
    ``fit`` via ``get_optional_runtime_spec_for_tensor_data``.

    Expected params:
        - ``strategy``: optional preprocessing strategy mapping, or
          ``None`` for runtime defaults / empty plan depending on ``auto``;
        - ``auto``: when ``True`` and ``strategy`` is ``None``, use
          runtime default steps; when ``False``, use an empty plan;
        - ``use_cache``: whether the optional service should use cache
          (default ``True``).
    """

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData) -> OptionalService:
        runtime_spec = get_optional_runtime_spec_for_tensor_data(train_data)

        strategy = self.params_for_fit.get('strategy')
        auto_select_strategy = self.params_for_fit.get('auto', True)
        if strategy is None:
            # Auto mode uses runtime defaults; otherwise keep an empty plan.
            strategy = (
                runtime_spec.default_steps if auto_select_strategy else {}
            )

        use_cache = bool(self.params_for_fit.get('use_cache', True))
        service = runtime_spec.service_cls(use_cache=use_cache)
        return service.fit(train_data, strategy)

    def predict(self, trained_operation: OptionalService, predict_data: TensorData) -> TensorData:
        return trained_operation.predict(predict_data)

    def predict_for_fit(self, trained_operation: OptionalService, predict_data: TensorData) -> TensorData:
        return self.predict(trained_operation, predict_data)
