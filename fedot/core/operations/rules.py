from typing import Any, FrozenSet, Mapping, Optional

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.preprocessor_types import (
    OptionalStepSpec,
    PreprocessingStepEnum,
    step_method_handlers,
)

OPTIONAL_NONE_METHOD_ALIASES = {'none', 'None', False}
OPTIONAL_AUTO_METHOD_ALIASES = {None, 'auto', 'Auto'}

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


def _optional_mapping_for_data_type(data_type: DataTypesEnum) -> Mapping:
    from fedot.preprocessing.tools.methods_mapping import (
        PREPROCESSING_OPTIONAL_MAPPING,
        TS_PREPROCESSING_MAPPING,
    )

    return (
        TS_PREPROCESSING_MAPPING
        if data_type == DataTypesEnum.ts
        else PREPROCESSING_OPTIONAL_MAPPING
    )


def _handler_methods_for_data_type(
    step: PreprocessingStepEnum,
    data_type: DataTypesEnum,
) -> Mapping[Any, Any]:
    return step_method_handlers(_optional_mapping_for_data_type(data_type).get(step))


def _method_names_from_mappings(step: PreprocessingStepEnum) -> set:
    names: set = set()
    for mapping in _optional_handler_mappings():
        for method in step_method_handlers(mapping.get(step)):
            names.add(normalize_optional_method_name(method))
    return names


def flat_optional_steps(data_type: DataTypesEnum) -> Mapping[PreprocessingStepEnum, bool]:
    """Flat knobs for ``data_type`` from that mapping's ``OptionalStepSpec``.

    Only steps with non-``None`` ``flat_auto_default`` expose ``<step>_method``.
    """
    steps: dict[PreprocessingStepEnum, bool] = {}
    for step, entry in _optional_mapping_for_data_type(data_type).items():
        if not isinstance(entry, OptionalStepSpec):
            continue
        if entry.flat_auto_default is None:
            continue
        steps[step] = entry.flat_auto_default
    return steps


def supported_optional_method_names(step: PreprocessingStepEnum) -> FrozenSet[Any]:
    """Flat-knob allowlist: auto/none + honest method values from handler mappings."""
    return frozenset({'auto', 'none'} | _method_names_from_mappings(step))


def resolve_optional_method(
    method: Any,
    data_type: DataTypesEnum,
    step: PreprocessingStepEnum,
):
    """Resolve a flat-knob / strategy method name to a handler-mapping enum for ``data_type``."""
    method_name = normalize_optional_method_name(method)
    if is_optional_auto_method(method_name) or is_optional_none_method(method_name):
        return method_name

    handlers = _handler_methods_for_data_type(step, data_type)
    for enum_key in handlers:
        if enum_key == method or normalize_optional_method_name(enum_key) == method_name:
            return enum_key

    raise KeyError(method_name)


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
    Mapped steps also accept ``auto`` / ``none``.
    """
    if step == PreprocessingStepEnum.custom:
        return None

    methods: set = set()
    for mapping in _optional_handler_mappings():
        step_methods = step_method_handlers(mapping.get(step))
        if not step_methods:
            continue
        methods.update(step_methods)
        methods.update(normalize_optional_method_name(method) for method in step_methods)

    methods.update(OPTIONAL_AUTO_METHOD_ALIASES)
    methods.update(OPTIONAL_NONE_METHOD_ALIASES)

    return frozenset(methods)
