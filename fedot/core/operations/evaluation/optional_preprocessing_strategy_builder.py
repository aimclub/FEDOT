from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.rules import (
    is_optional_auto_method,
    is_optional_none_method,
    resolve_optional_imputation_method,
    resolve_optional_scaling_method,
)
from fedot.core.operations.schemas import (
    validate_optional_imputation_method,
    validate_optional_scaling_method,
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
    use_imputation = params.get('use_imputation')
    use_scaling = params.get('use_scaling')

    if auto is False and use_imputation is None and use_scaling is None:
        return {}

    if use_imputation is None:
        use_imputation = True
    if use_scaling is None:
        use_scaling = True

    imputation_method = validate_optional_imputation_method(
        params.get('imputation_method', 'auto')
    )
    scaling_method = validate_optional_scaling_method(
        params.get('scaling_method', 'auto')
    )

    strategy: Dict[PreprocessingStepEnum, Any] = {}

    if use_imputation:
        resolved = resolve_optional_imputation_method(imputation_method, data.data_type)
        if not is_optional_none_method(resolved):
            strategy[PreprocessingStepEnum.imputation] = (
                None if is_optional_auto_method(resolved) else resolved
            )

    if use_scaling:
        resolved = resolve_optional_scaling_method(scaling_method, data.data_type)
        if not is_optional_none_method(resolved):
            strategy[PreprocessingStepEnum.scaling] = (
                None if is_optional_auto_method(resolved) else resolved
            )

    return strategy


def build_optional_strategy_from_node_params(
    data: TensorData,
    params: Union[OperationParameters, Dict[str, Any], None],
    context: Optional[ValidationContext] = None,
) -> OptionalStrategySpec:
    """Build a validated OptionalStrategySpec from flat knobs or strategy override.

    Supported params:
        - ``strategy``: full override mapping for OptionalService;
        - ``auto``: when ``False`` and step flags are unset, returns empty plan;
        - ``use_imputation`` / ``imputation_method``;
        - ``use_scaling`` / ``scaling_method``.

    Method values:
        - ``auto`` → planner default step creation (``None`` stage config);
        - ``none`` → skip stage;
        - concrete names (``mean``, ``standard``, ...) → method-only stage config;
          planner selects columns automatically.

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
