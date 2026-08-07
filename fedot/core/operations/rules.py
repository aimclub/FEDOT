from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    PreprocessingStepEnum,
    ScalingMethodEnum,
)

OPTIONAL_NONE_METHOD_ALIASES = {'none', 'None', False}
OPTIONAL_AUTO_METHOD_ALIASES = {None, 'auto', 'Auto'}

# Short flat-knob names that remap by data_type (name != final enum.value or differs for TS).
# Everything else is resolved directly from handler mapping method values.
TABULAR_OPTIONAL_IMPUTATION_ALIASES: Dict[str, ImputationMethodEnum] = {
    'mean': ImputationMethodEnum.mean,
    'median': ImputationMethodEnum.median,
    'mode': ImputationMethodEnum.mode,
}

TS_OPTIONAL_IMPUTATION_ALIASES: Dict[str, ImputationMethodEnum] = {
    'mean': ImputationMethodEnum.ts_mean,
    'median': ImputationMethodEnum.ts_median,
    'mode': ImputationMethodEnum.ts_mean,
}

TABULAR_OPTIONAL_SCALING_ALIASES: Dict[str, ScalingMethodEnum] = {
    'standard': ScalingMethodEnum.standard,
    'min_max': ScalingMethodEnum.min_max,
    'robust': ScalingMethodEnum.robust,
}

TS_OPTIONAL_SCALING_ALIASES: Dict[str, ScalingMethodEnum] = {
    'standard': ScalingMethodEnum.standart_per_channel,
    'min_max': ScalingMethodEnum.seasonal,
    'robust': ScalingMethodEnum.rolling,
}

OPTIONAL_STRATEGY_STAGE_PARAM_KEYS: FrozenSet[str] = frozenset({
    'method',
    'features_idx',
    'step_args',
    'implementation',
})


def normalize_optional_method_name(method: Any) -> Any:
    if hasattr(method, 'value'):
        return method.value
    return method


def is_optional_auto_method(method: Any) -> bool:
    return normalize_optional_method_name(method) in OPTIONAL_AUTO_METHOD_ALIASES


def is_optional_none_method(method: Any) -> bool:
    return normalize_optional_method_name(method) in OPTIONAL_NONE_METHOD_ALIASES


def _optional_handler_mappings():
    from fedot.preprocessing.tools.methods_mapping import (
        PREPROCESSING_OPTIONAL_MAPPING,
        TS_PREPROCESSING_MAPPING,
    )

    return (PREPROCESSING_OPTIONAL_MAPPING, TS_PREPROCESSING_MAPPING)


def _handler_methods_for_data_type(
    step: PreprocessingStepEnum,
    data_type: DataTypesEnum,
) -> Mapping[Any, Any]:
    from fedot.preprocessing.tools.methods_mapping import (
        PREPROCESSING_OPTIONAL_MAPPING,
        TS_PREPROCESSING_MAPPING,
    )

    mapping = (
        TS_PREPROCESSING_MAPPING
        if data_type == DataTypesEnum.ts
        else PREPROCESSING_OPTIONAL_MAPPING
    )
    return mapping.get(step, {})


def _method_names_from_mappings(step: PreprocessingStepEnum) -> set:
    names: set = set()
    for mapping in _optional_handler_mappings():
        for method in mapping.get(step, {}):
            names.add(normalize_optional_method_name(method))
    return names


def _flat_alias_names(alias_tables: Iterable[Mapping[str, Any]]) -> set:
    names: set = set()
    for table in alias_tables:
        names.update(table)
    return names


def supported_optional_imputation_method_names() -> FrozenSet[Any]:
    """Flat-knob allowlist: auto/none + aliases + method values from handler mappings."""
    return frozenset(
        {'auto', 'none'}
        | _flat_alias_names((TABULAR_OPTIONAL_IMPUTATION_ALIASES, TS_OPTIONAL_IMPUTATION_ALIASES))
        | _method_names_from_mappings(PreprocessingStepEnum.imputation)
    )


def supported_optional_scaling_method_names() -> FrozenSet[Any]:
    """Flat-knob allowlist: auto/none + aliases + method values from handler mappings."""
    return frozenset(
        {'auto', 'none'}
        | _flat_alias_names((TABULAR_OPTIONAL_SCALING_ALIASES, TS_OPTIONAL_SCALING_ALIASES))
        | _method_names_from_mappings(PreprocessingStepEnum.scaling)
    )


def _resolve_from_aliases_or_mapping(
    method: Any,
    data_type: DataTypesEnum,
    *,
    step: PreprocessingStepEnum,
    aliases_by_data_type: Mapping[DataTypesEnum, Mapping[str, Any]],
):
    method_name = normalize_optional_method_name(method)
    if is_optional_auto_method(method_name) or is_optional_none_method(method_name):
        return method_name

    aliases = aliases_by_data_type.get(data_type, {})
    if method_name in aliases:
        return aliases[method_name]

    handlers = _handler_methods_for_data_type(step, data_type)
    for enum_key in handlers:
        if enum_key == method or normalize_optional_method_name(enum_key) == method_name:
            return enum_key

    raise KeyError(method_name)


def resolve_optional_imputation_method(method: Any, data_type: DataTypesEnum):
    return _resolve_from_aliases_or_mapping(
        method,
        data_type,
        step=PreprocessingStepEnum.imputation,
        aliases_by_data_type={
            DataTypesEnum.tabular: TABULAR_OPTIONAL_IMPUTATION_ALIASES,
            DataTypesEnum.ts: TS_OPTIONAL_IMPUTATION_ALIASES,
        },
    )


def resolve_optional_scaling_method(method: Any, data_type: DataTypesEnum):
    return _resolve_from_aliases_or_mapping(
        method,
        data_type,
        step=PreprocessingStepEnum.scaling,
        aliases_by_data_type={
            DataTypesEnum.tabular: TABULAR_OPTIONAL_SCALING_ALIASES,
            DataTypesEnum.ts: TS_OPTIONAL_SCALING_ALIASES,
        },
    )


def supported_optional_strategy_steps() -> FrozenSet[PreprocessingStepEnum]:
    """Steps allowed in optional strategy = keys of handler mappings + custom."""
    steps = {PreprocessingStepEnum.custom}
    for mapping in _optional_handler_mappings():
        steps.update(mapping)
    return frozenset(steps)


def allowed_optional_strategy_methods(
    step: PreprocessingStepEnum,
) -> Optional[FrozenSet[Any]]:
    """Methods allowed for a step, derived from handler mapping keys.

    ``custom`` accepts any method (implementation is provided by the caller).
    Imputation/scaling also accept ``auto`` / ``none`` and flat-knob aliases.
    """
    if step == PreprocessingStepEnum.custom:
        return None

    methods: set = set()
    for mapping in _optional_handler_mappings():
        step_methods = mapping.get(step)
        if not step_methods:
            continue
        methods.update(step_methods)
        methods.update(normalize_optional_method_name(method) for method in step_methods)

    if step == PreprocessingStepEnum.imputation:
        methods.update(OPTIONAL_AUTO_METHOD_ALIASES)
        methods.update(OPTIONAL_NONE_METHOD_ALIASES)
        methods.update(_flat_alias_names(
            (TABULAR_OPTIONAL_IMPUTATION_ALIASES, TS_OPTIONAL_IMPUTATION_ALIASES)
        ))
    elif step == PreprocessingStepEnum.scaling:
        methods.update(OPTIONAL_AUTO_METHOD_ALIASES)
        methods.update(OPTIONAL_NONE_METHOD_ALIASES)
        methods.update(_flat_alias_names(
            (TABULAR_OPTIONAL_SCALING_ALIASES, TS_OPTIONAL_SCALING_ALIASES)
        ))

    return frozenset(methods)
