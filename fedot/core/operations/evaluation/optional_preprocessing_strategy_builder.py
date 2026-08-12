from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.rules import (
    FLAT_OPTIONAL_STEPS,
    is_optional_auto_method,
    is_optional_none_method,
    resolve_optional_method,
)
from fedot.core.operations.schemas import (
    validate_optional_method,
    validate_optional_strategy_mapping,
)
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.validation.context import ValidationContext


@dataclass(frozen=True)
class OptionalStrategySpec(Mapping[PreprocessingStepEnum, Any]):
    """Validated optional-preprocessing strategy for all node entry paths."""

    steps: Mapping[PreprocessingStepEnum, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'steps', MappingProxyType(dict(self.steps)))

    def __getitem__(self, key: PreprocessingStepEnum) -> Any:
        return self.steps[key]

    def __iter__(self) -> Iterator[PreprocessingStepEnum]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


def _build_strategy_from_flat_params(
    data: TensorData,
    params: OperationParameters,
) -> Dict[PreprocessingStepEnum, Any]:
    auto = params.get('auto', True)
    strategy: Dict[PreprocessingStepEnum, Any] = {}

    for step, default_enabled in FLAT_OPTIONAL_STEPS.items():
        method = params.get(f'{step.value}_method')
        if method is None:
            if auto and default_enabled:
                method = 'auto'
            else:
                continue

        method = validate_optional_method(step, method)
        resolved = resolve_optional_method(method, data.data_type, step)
        if is_optional_none_method(resolved):
            continue
        strategy[step] = None if is_optional_auto_method(resolved) else resolved

    return strategy


def build_optional_strategy_from_node_params(
    data: TensorData,
    params: Union[OperationParameters, Dict[str, Any], None],
    context: Optional[ValidationContext] = None,
) -> OptionalStrategySpec:
    """Build a validated OptionalStrategySpec from flat knobs or strategy override.

    Supported params:
        - ``strategy``: full override mapping for OptionalService;
        - ``auto``: when ``True`` (default), unset methods follow ``FLAT_OPTIONAL_STEPS``
          (enabled steps get ``auto``); when ``False``, unset methods are skipped;
        - ``<step>_method`` for steps in ``FLAT_OPTIONAL_STEPS``
          (imputation, scaling, filtering, ...).

    Method values:
        - unset → skip (unless ``auto=True`` and step default-enabled → ``auto``);
        - ``auto`` → planner default step creation (``None`` stage config);
        - ``none`` → skip stage;
        - concrete names from handler mapping for the data type
          (``mean``, ``ts_mean``, ``standard``, ``seasonal``, ...) → method-only
          stage config; planner selects columns automatically.

    Both entry paths go through the same fail-fast strategy validation.
    """
    if params is None:
        params = OperationParameters()
    elif isinstance(params, dict):
        params = OperationParameters(**params)

    explicit_strategy = params.get('strategy')
    if explicit_strategy is not None:
        raw_strategy = explicit_strategy
    else:
        raw_strategy = _build_strategy_from_flat_params(data, params)

    normalized = validate_optional_strategy_mapping(
        raw_strategy,
        data.data_type,
        context=context,
    )
    return OptionalStrategySpec(steps=normalized)
