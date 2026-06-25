from typing import Optional

import torch
import torch.nn as nn

from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.operation_implementations.models.torch import TorchLinearClassifier


class TorchTabularClassificationStrategy(EvaluationStrategy):
    """TensorData-native classification strategy without numpy/sklearn bridge."""

    _operations_by_types = {
        'torch_linear': TorchLinearClassifier,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, features: torch.Tensor, target: torch.Tensor):
        operation_implementation = self.operation_impl(self.params_for_fit)
        operation_implementation.fit(features, target)
        return operation_implementation

    def predict(self, trained_operation, features: torch.Tensor) -> torch.Tensor:
        if self.output_mode == 'labels':
            prediction = trained_operation.predict_labels(features)
        elif self.output_mode in ['probs', 'full_probs', 'default', False]:
            prediction = trained_operation.predict_proba(features)
            if self.output_mode != 'full_probs' and prediction.shape[-1] == 2:
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return prediction
