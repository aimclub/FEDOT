"""Runtime lookup and single-call interpreters for validated extension specs."""
from typing import Any, Dict, Optional

import numpy as np
import torch
from pymonad.either import Left, Right

from fedot.extensions.call_rules import invoke_factory, invoke_once
from fedot.extensions.contracts import (
    ExtensionContractError, ExtensionError, ExternalModelSpec, ExternalTransformSpec,
    unwrap_extension_result,
)
from fedot.extensions.data_type_rules import build_extension_data_type_view
from fedot.extensions.parameter_rules import extract_factory_params, resolve_extension_params
from fedot.extensions.registry import get_registered_extensions
from fedot.extensions.runtime_contracts import (
    ExternalModelImplementation, ModelInput, ModelOutput, TransformImplementation, TransformInput, TransformOutput,
)


def get_extension_operation_spec(operation_name):
    operation_name = operation_name.split('/')[0]
    for extension in get_registered_extensions():
        for spec in extension.manifest.models + extension.manifest.transforms:
            if spec.name == operation_name:
                return spec
    return None


def get_extension_model_spec(operation_name: str) -> Optional[ExternalModelSpec]:
    spec = get_extension_operation_spec(operation_name)
    return spec if isinstance(spec, ExternalModelSpec) else None


def get_extension_transform_spec(operation_name: str) -> Optional[ExternalTransformSpec]:
    spec = get_extension_operation_spec(operation_name)
    return spec if isinstance(spec, ExternalTransformSpec) else None


def is_extension_operation_name(operation_name: str) -> bool:
    return get_extension_operation_spec(operation_name) is not None


def require_extension_spec(operation_name):
    spec = get_extension_operation_spec(operation_name)
    if spec is None:
        raise ExtensionContractError(ExtensionError(
            'operation_not_registered', f'Extension operation "{operation_name}" is not registered.'))
    return spec


def try_build_extension_strategy_params(operation_name: str,
                                        user_params: Optional[Dict[str, Any]] = None,
                                        output_mode: str = 'default'):
    model_spec = get_extension_model_spec(operation_name)
    if model_spec is None:
        return Left(ExtensionError('operation_not_registered',
                                   f'Extension model "{operation_name}" is not registered.'))
    resolution = resolve_extension_params(model_spec, user_params)
    if resolution.is_left():
        return resolution
    return Right({
        **resolution.value,
        'model_fit': _build_model_fit(model_spec),
        'model_predict': _build_model_predict(model_spec),
        '_extension_output_mode': output_mode,
    })


def build_extension_strategy_params(operation_name: str,
                                    user_params: Optional[Dict[str, Any]] = None,
                                    output_mode: str = 'default') -> Dict[str, Any]:
    return unwrap_extension_result(try_build_extension_strategy_params(
        operation_name, user_params, output_mode))


def get_extension_acceptable_task_types(operation_name: str):
    return require_extension_spec(operation_name).capabilities.tasks


def get_extension_data_types(operation_name: str):
    spec = require_extension_spec(operation_name)
    return build_extension_data_type_view(spec.capabilities.data_types).input_types


def get_extension_tensor_data_types(operation_name: str):
    spec = require_extension_spec(operation_name)
    return build_extension_data_type_view(spec.capabilities.data_types).tensor_types


def _method(instance, name):
    method = getattr(instance, name, None)
    if not callable(method):
        raise ExtensionContractError(ExtensionError(
            'missing_runtime_method', f'Extension must define callable {name}.', {'method': name}))
    return method


def _instantiate_model(spec, params):
    resolved = unwrap_extension_result(resolve_extension_params(spec, params))
    instance = invoke_factory(spec.factory, resolved)
    if instance is None:
        raise ExtensionContractError(ExtensionError('factory_returned_none', 'Factory returned None.'))
    return instance


def _call_with_supported_signature(method, *candidate_args):
    return invoke_once(method, tuple((args, {}) for args in candidate_args))


def _fit_instance(instance, data, params):
    _call_with_supported_signature(
        _method(instance, 'fit'), (data.features, data.target), (data.features,),
        (data.idx, data.features, data.target, params), (data.idx, data.features, data.target))


def _check_target(spec, data):
    if spec.capabilities.requires_target and data.target is None:
        raise ExtensionContractError(ExtensionError('target_required', 'Operation requires a training target.'))


def fit_model(spec: ExternalModelSpec, data: ModelInput, params) -> ExternalModelImplementation:
    _check_target(spec, data)
    instance = _instantiate_model(spec, params)
    if not callable(getattr(instance, 'predict', None)):
        _method(instance, 'predict_proba')
    _fit_instance(instance, data, params)
    return instance


def _validate_array(value, n_rows, field):
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise ExtensionContractError(ExtensionError(
            'invalid_runtime_output', f'{field} must be a numpy array or torch tensor.'))
    if value.ndim == 0 or value.shape[0] != n_rows:
        raise ExtensionContractError(ExtensionError(
            'output_row_mismatch', f'{field} must preserve sample count.'))
    return value


def predict_model(spec: ExternalModelSpec, instance, data: ModelInput, params,
                  output_mode='default') -> ModelOutput:
    if output_mode not in ('default', 'labels', 'probs', 'full_probs'):
        raise ExtensionContractError(ExtensionError('unsupported_output_mode', 'Unsupported model output mode.'))
    proba = callable(getattr(instance, 'predict_proba', None))
    if output_mode in ('probs', 'full_probs') and not proba:
        raise ExtensionContractError(ExtensionError('unsupported_output_mode', 'Model has no predict_proba.'))
    use_proba = proba and output_mode in ('default', 'probs', 'full_probs')
    method = _method(instance, 'predict_proba' if use_proba else 'predict')
    prediction = _call_with_supported_signature(
        method, (data.features,), (data.idx, data.features, params), (data.idx, data.features))
    prediction = _validate_array(prediction, len(data.features), 'prediction')
    if use_proba and output_mode != 'full_probs' and prediction.ndim == 2 and prediction.shape[1] == 2:
        prediction = prediction[:, 1]
    return ModelOutput(prediction)


def fit_transformer(spec: ExternalTransformSpec, data: TransformInput, params) -> TransformImplementation:
    _check_target(spec, data)
    instance = _instantiate_model(spec, params)
    _method(instance, 'transform')
    if spec.capabilities.requires_fit:
        _fit_instance(instance, data, params)
    return instance


def transform_features(spec: ExternalTransformSpec, instance, data: TransformInput, params) -> TransformOutput:
    features = _call_with_supported_signature(
        _method(instance, 'transform'), (data.features,), (data.idx, data.features, params),
        (data.idx, data.features))
    features = _validate_array(features, len(data.features), 'features')
    if features.ndim not in (1, 2, 3) or any(size == 0 for size in features.shape):
        raise ExtensionContractError(ExtensionError('invalid_runtime_output', 'Invalid transformed feature axes.'))
    return TransformOutput(features)


def _build_model_fit(model_spec: ExternalModelSpec):
    def _fit(idx, features, target, params):
        return fit_model(model_spec, ModelInput(features, target, idx), extract_factory_params(params))
    return _fit


def _build_model_predict(model_spec: ExternalModelSpec):
    def _predict(fitted_model, idx, features, params):
        if fitted_model is None:
            raise ExtensionContractError(ExtensionError('operation_not_fitted', 'Model must be fitted first.'))
        output = predict_model(model_spec, fitted_model, ModelInput(features, idx=idx),
                               extract_factory_params(params), params.get('_extension_output_mode', 'default'))
        return output.prediction, _infer_output_type_name(model_spec)
    return _predict


def _infer_output_type_name(model_spec: ExternalModelSpec) -> str:
    output_type = model_spec.capabilities.output_data_type
    types = (output_type,) if output_type is not None else model_spec.capabilities.data_types
    return build_extension_data_type_view(types).preferred_output_type_name
