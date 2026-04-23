import inspect
from typing import Any, Dict, Optional

from pymonad.either import Right

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.extensions.contracts import ExternalModelSpec
from fedot.extensions.data_type_rules import build_extension_data_type_view
from fedot.extensions.parameter_rules import extract_factory_params, resolve_extension_params
from fedot.extensions.registry import get_registered_extensions


def get_extension_model_spec(operation_name: str) -> Optional[ExternalModelSpec]:
    for registered_extension in get_registered_extensions():
        for model in registered_extension.manifest.models:
            if model.name == operation_name:
                return model
    return None


def is_extension_operation_name(operation_name: str) -> bool:
    return get_extension_model_spec(operation_name) is not None


def try_build_extension_strategy_params(operation_name: str,
                                        user_params: Optional[Dict[str, Any]] = None,
                                        output_mode: str = 'default'):
    model_spec = get_extension_model_spec(operation_name)
    if model_spec is None:
        raise ValueError(f'Extension model "{operation_name}" is not registered.')

    params_resolution = resolve_extension_params(model_spec, user_params)
    if params_resolution.is_left():
        return params_resolution

    resolved_user_params = params_resolution.value
    return Right({
        **resolved_user_params,
        'model_fit': _build_model_fit(model_spec),
        'model_predict': _build_model_predict(model_spec),
        '_extension_output_mode': output_mode,
    })


def build_extension_strategy_params(operation_name: str,
                                    user_params: Optional[Dict[str, Any]] = None,
                                    output_mode: str = 'default') -> Dict[str, Any]:
    strategy_params = try_build_extension_strategy_params(operation_name, user_params, output_mode)
    if strategy_params.is_left():
        raise ValueError(strategy_params.monoid[0].message)
    return strategy_params.value


def get_extension_acceptable_task_types(operation_name: str):
    model_spec = get_extension_model_spec(operation_name)
    if model_spec is None:
        raise ValueError(f'Extension model "{operation_name}" is not registered.')
    return model_spec.capabilities.tasks


def get_extension_data_types(operation_name: str):
    model_spec = get_extension_model_spec(operation_name)
    if model_spec is None:
        raise ValueError(f'Extension model "{operation_name}" is not registered.')
    return build_extension_data_type_view(model_spec.capabilities.data_types).input_types


def get_extension_tensor_data_types(operation_name: str):
    model_spec = get_extension_model_spec(operation_name)
    if model_spec is None:
        raise ValueError(f'Extension model "{operation_name}" is not registered.')
    return build_extension_data_type_view(model_spec.capabilities.data_types).tensor_types


def _build_model_fit(model_spec: ExternalModelSpec):
    def _fit(idx, features, target, params):
        model = _instantiate_model(model_spec, params)
        fit_method = getattr(model, 'fit', None)
        if callable(fit_method):
            _call_with_supported_signature(
                fit_method,
                (features, target),
                (features,),
                (idx, features, target, params),
                (idx, features, target),
            )
        return model

    return _fit


def _build_model_predict(model_spec: ExternalModelSpec):
    def _predict(fitted_model, idx, features, params):
        model = fitted_model if fitted_model is not None else _instantiate_model(model_spec, params)
        output_mode = params.get('_extension_output_mode', 'default')

        if output_mode in ('probs', 'full_probs', 'default') and hasattr(model, 'predict_proba'):
            prediction = _call_with_supported_signature(
                getattr(model, 'predict_proba'),
                (features,),
                (idx, features, params),
                (idx, features),
            )
            if output_mode != 'full_probs' and getattr(
                    prediction, 'shape', None) is not None and len(
                    prediction.shape) > 1 and prediction.shape[1] == 2:
                prediction = prediction[:, 1]
        elif hasattr(model, 'predict'):
            prediction = _call_with_supported_signature(
                getattr(model, 'predict'),
                (features,),
                (idx, features, params),
                (idx, features),
            )
        elif hasattr(model, 'transform'):
            prediction = _call_with_supported_signature(
                getattr(model, 'transform'),
                (features,),
                (idx, features, params),
                (idx, features),
            )
        else:
            raise TypeError(f'Extension model "{model_spec.name}" must define predict, predict_proba, or transform.')

        output_type = _infer_output_type_name(model_spec)
        return prediction, output_type

    return _predict


def _instantiate_model(model_spec: ExternalModelSpec, params: Dict[str, Any]):
    factory = model_spec.factory
    user_params = extract_factory_params(params)
    try:
        signature = inspect.signature(factory)
        signature.bind_partial(user_params)
        return factory(user_params)
    except TypeError:
        return factory()


def _call_with_supported_signature(method, *candidate_args):
    signature = inspect.signature(method)
    last_error = None
    for args in candidate_args:
        try:
            signature.bind_partial(*args)
            return method(*args)
        except TypeError as error:
            last_error = error
            continue
    raise last_error or TypeError('No supported signature found for extension model method.')


def _infer_output_type_name(model_spec: ExternalModelSpec) -> str:
    return build_extension_data_type_view(model_spec.capabilities.data_types).preferred_output_type_name
