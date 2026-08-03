from typing import Any, Dict

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    ScalingMethodEnum,
)

OPTIONAL_NONE_METHOD_ALIASES = {'none', 'None', False}
OPTIONAL_AUTO_METHOD_ALIASES = {None, 'auto', 'Auto'}

TABULAR_OPTIONAL_IMPUTATION_METHODS: Dict[str, ImputationMethodEnum] = {
    'mean': ImputationMethodEnum.mean,
    'median': ImputationMethodEnum.median,
    'mode': ImputationMethodEnum.mode,
}

TS_OPTIONAL_IMPUTATION_METHODS: Dict[str, ImputationMethodEnum] = {
    'mean': ImputationMethodEnum.ts_mean,
    'median': ImputationMethodEnum.ts_median,
    'mode': ImputationMethodEnum.ts_mean,
}

TABULAR_OPTIONAL_SCALING_METHODS: Dict[str, ScalingMethodEnum] = {
    'standard': ScalingMethodEnum.standard,
    'min_max': ScalingMethodEnum.min_max,
    'robust': ScalingMethodEnum.robust,
}

TS_OPTIONAL_SCALING_METHODS: Dict[str, ScalingMethodEnum] = {
    'standard': ScalingMethodEnum.standart_per_channel,
    'min_max': ScalingMethodEnum.seasonal,
    'robust': ScalingMethodEnum.rolling,
}

SUPPORTED_OPTIONAL_IMPUTATION_METHOD_NAMES = (
    'auto',
    'none',
    *TABULAR_OPTIONAL_IMPUTATION_METHODS.keys(),
)

SUPPORTED_OPTIONAL_SCALING_METHOD_NAMES = (
    'auto',
    'none',
    *TABULAR_OPTIONAL_SCALING_METHODS.keys(),
)


def normalize_optional_method_name(method: Any) -> Any:
    if hasattr(method, 'value'):
        return method.value
    return method


def is_optional_auto_method(method: Any) -> bool:
    return normalize_optional_method_name(method) in OPTIONAL_AUTO_METHOD_ALIASES


def is_optional_none_method(method: Any) -> bool:
    return normalize_optional_method_name(method) in OPTIONAL_NONE_METHOD_ALIASES


def resolve_optional_imputation_method(method: Any, data_type: DataTypesEnum):
    method_name = normalize_optional_method_name(method)
    if is_optional_auto_method(method_name) or is_optional_none_method(method_name):
        return method_name
    mapping = (
        TS_OPTIONAL_IMPUTATION_METHODS
        if data_type == DataTypesEnum.ts
        else TABULAR_OPTIONAL_IMPUTATION_METHODS
    )
    return mapping[method_name]


def resolve_optional_scaling_method(method: Any, data_type: DataTypesEnum):
    method_name = normalize_optional_method_name(method)
    if is_optional_auto_method(method_name) or is_optional_none_method(method_name):
        return method_name
    mapping = (
        TS_OPTIONAL_SCALING_METHODS
        if data_type == DataTypesEnum.ts
        else TABULAR_OPTIONAL_SCALING_METHODS
    )
    return mapping[method_name]
