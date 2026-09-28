"""Thin TensorData interpreters for model and transform extension contracts."""
from dataclasses import replace

import numpy as np
import torch

from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.extensions.contracts import ArrayBackend, ExtensionContractError, ExtensionError, unwrap_extension_result
from fedot.extensions.execution_rules import plan_execution
from fedot.extensions.parameter_rules import resolve_extension_params
from fedot.extensions.runtime_contracts import ModelInput, TransformInput
from fedot.extensions.runtime_rules import (
    fit_model, fit_transformer, predict_model, require_extension_spec, transform_features,
)


def _input_array(value, backend):
    if value is None:
        return None
    # External code owns private buffers, never the caller's TensorData storage.
    if backend is ArrayBackend.numpy:
        return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
    return value.clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value).clone()


def _output_tensor(value, data):
    try:
        return torch.as_tensor(value, device=data.features.device).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ExtensionContractError(ExtensionError(
            'invalid_runtime_output', 'Output cannot be represented as a tensor.', cause=exc)) from exc


class _ExtensionStrategy(EvaluationStrategy):
    def __init__(self, operation_type, params=None):
        super().__init__(operation_type, params)
        self.spec = require_extension_spec(operation_type)
        self.params = unwrap_extension_result(resolve_extension_params(self.spec, self.params_for_fit.to_dict()))
        self.output_mode = 'default'

    def _prepare(self, data, input_type, fitting=False):
        plan = unwrap_extension_result(plan_execution(
            self.spec, data.task.task_type, data.data_type,
            fitting=fitting, has_target=data.target is not None))
        request = input_type(_input_array(data.features, plan.backend),
                             _input_array(data.target, plan.backend),
                             _input_array(data.idx, ArrayBackend.numpy))
        return plan, request


class ExtensionModelStrategy(_ExtensionStrategy):
    def fit(self, train_data):
        _, request = self._prepare(train_data, ModelInput, fitting=True)
        return fit_model(self.spec, request, self.params)

    def predict(self, trained_operation, predict_data):
        plan, request = self._prepare(predict_data, ModelInput)
        output = predict_model(self.spec, trained_operation, request, self.params, self.output_mode)
        return replace(predict_data, predict=_output_tensor(output.prediction, predict_data),
                       data_type=plan.output_data_type)


class ExtensionTransformStrategy(_ExtensionStrategy):
    def fit(self, train_data):
        _, request = self._prepare(train_data, TransformInput, fitting=True)
        return fit_transformer(self.spec, request, self.params)

    def predict(self, trained_operation, predict_data):
        plan, request = self._prepare(predict_data, TransformInput)
        output = transform_features(self.spec, trained_operation, request, self.params)
        features = _output_tensor(output.features, predict_data)
        # Column semantics belong to the transform; do not retain stale source selectors.
        return replace(predict_data, features=features, predict=None, data_type=plan.output_data_type,
                       features_names=None, categorical_idx=[], numerical_idx=[], idx_mapping={})
