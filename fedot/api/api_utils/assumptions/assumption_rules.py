from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from fedot.core.constants import AUTO_PRESET_NAME, BEST_QUALITY_PRESET_NAME
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import RepositoryKind


@dataclass(frozen=True)
class AssumptionsFilterDecision:
    allow_filtering: bool
    whitelist: Tuple[str, ...]
    sampling_choices: Tuple[str, ...]


@dataclass(frozen=True)
class PresetSpec:
    requested_preset: str
    base_preset: str
    modification: Optional[str]
    use_auto: bool
    use_stable: bool
    use_gpu: bool


_REQUIRED_SOURCE_OPERATIONS = {
    DataTypesEnum.image: ('data_source_img',),
    DataTypesEnum.text: ('data_source_text',),
    DataTypesEnum.table: ('data_source_table',),
}


def default_repository_name_for_data(data) -> str:
    if data.data_type == DataTypesEnum.multi_ts:
        return RepositoryKind.all.value
    return RepositoryKind.model.value


def required_operations_for_data(data, data_type: DataTypesEnum) -> Tuple[str, ...]:
    required_operations = []

    if hasattr(data, 'items'):
        required_operations.extend(_REQUIRED_SOURCE_OPERATIONS.get(data_type, ()))

    if data_type is DataTypesEnum.image:
        required_operations.append('cnn')

    return tuple(dict.fromkeys(required_operations))


def build_operations_filter_decision(data,
                                     data_type: DataTypesEnum,
                                     available_operations: Optional[Sequence[str]],
                                     suitable_operations: Iterable[str]) -> AssumptionsFilterDecision:
    whitelist = tuple(dict.fromkeys(available_operations or ()))
    suitable_set = set(suitable_operations)
    sampling_choices = [operation for operation in whitelist if operation in suitable_set]

    for required_operation in required_operations_for_data(data, data_type):
        if required_operation not in sampling_choices:
            sampling_choices.append(required_operation)

    return AssumptionsFilterDecision(
        allow_filtering=bool(sampling_choices),
        whitelist=whitelist,
        sampling_choices=tuple(sampling_choices),
    )


def parse_preset_spec(preset_name: Optional[str]) -> PresetSpec:
    requested_preset = preset_name or ''
    base_preset = requested_preset
    modification = None
    use_auto = AUTO_PRESET_NAME in requested_preset
    use_stable = 'stable' in requested_preset
    use_gpu = 'gpu' in requested_preset

    if '*' in base_preset:
        base_name, suffix = base_preset.split('*', 1)
        base_preset = base_name
        modification = f'*{suffix}'

    if use_stable:
        base_preset = BEST_QUALITY_PRESET_NAME

    return PresetSpec(
        requested_preset=requested_preset,
        base_preset=base_preset,
        modification=modification,
        use_auto=use_auto,
        use_stable=use_stable,
        use_gpu=use_gpu,
    )


def merge_preset_operations(base_operations: Iterable[str],
                            modification_operations: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    merged_operations = list(dict.fromkeys(base_operations))
    if modification_operations is None:
        return tuple(merged_operations)

    modification_set = set(modification_operations)
    return tuple(operation for operation in merged_operations if operation in modification_set)


def exclude_operations(available_operations: Iterable[str], excluded_operations: Iterable[str]) -> Tuple[str, ...]:
    excluded_set = set(excluded_operations)
    return tuple(operation for operation in available_operations if operation not in excluded_set)


def finalize_operations(available_operations: Iterable[str], excluded_operations: Iterable[str] = ()) -> list[str]:
    return sorted(set(exclude_operations(available_operations, excluded_operations)))
