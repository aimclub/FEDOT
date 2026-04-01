from __future__ import annotations

import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.industrial_nn_bridge_rules import (
    build_industrial_nn_bridge_params,
    normalize_industrial_nn_prediction,
    resolve_industrial_nn_model_class,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class IndustrialNNBridgeImplementation:
    """Thin runtime adapter over industrial neural models for benchmark-only core operations."""

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_type = operation_type
        self.params = params or OperationParameters()
        self._fitted_model = None

    def fit(self, input_data: InputData):
        model_class = resolve_industrial_nn_model_class(self.operation_type)
        model_params = build_industrial_nn_bridge_params(
            operation_type=self.operation_type,
            user_params=self.params.to_dict(),
        )
        model = model_class(OperationParameters(**model_params))
        self._fitted_model = model.fit(input_data)
        return self

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        if self._fitted_model is None:
            raise ValueError('Industrial NN bridge model is not fitted yet')

        prediction = self._fitted_model.predict(input_data, output_mode='default')
        return normalize_industrial_nn_prediction(
            prediction=prediction,
            reference_data=input_data,
            output_mode=output_mode,
        )

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.predict(input_data=input_data, output_mode=output_mode)


class IndustrialNNBridgeStrategy(EvaluationStrategy):
    """Benchmark-only strategy for opt-in industrial neural bridge operations."""

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = IndustrialNNBridgeImplementation(operation_type, params)

    def fit(self, train_data: InputData):
        with ImplementationRandomStateHandler(implementation=self.operation_impl):
            self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data, output_mode=self.output_mode)
        return self._convert_to_output(prediction, predict_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict_for_fit(predict_data, output_mode=self.output_mode)
        return self._convert_to_output(prediction, predict_data)

    def _convert_to_operation(self, operation_type: str):
        return IndustrialNNBridgeImplementation
