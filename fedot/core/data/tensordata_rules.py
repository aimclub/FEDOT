from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

SUPPORTED_BACKEND_NAMES = ('cpu', 'gpu')

LEGACY_DATA_TYPE_MAPPING = {
    DataTypesEnum.tabular: DataTypesEnum.tabular,
    DataTypesEnum.table: DataTypesEnum.tabular,
    DataTypesEnum.ts: DataTypesEnum.ts,
    DataTypesEnum.multi_ts: DataTypesEnum.ts,
    DataTypesEnum.text: DataTypesEnum.tabular,
    DataTypesEnum.image: DataTypesEnum.ts,
    'tabular': DataTypesEnum.tabular,
    'table': DataTypesEnum.tabular,
    'ts': DataTypesEnum.ts,
    'time_series': DataTypesEnum.ts,
    'multi_ts': DataTypesEnum.ts,
    'multi_time_series': DataTypesEnum.ts,
    'text': DataTypesEnum.tabular,
    'image': DataTypesEnum.ts,
}


@dataclass(frozen=True)
class TensorDataIdentity:
    task: Task
    data_type: DataTypesEnum
    state: StateEnum


@dataclass(frozen=True)
class TensorDataBackendPlan:
    backend_name: str


@dataclass(frozen=True)
class TensorDataDeviceSyncPlan:
    should_move_to_backend: bool


@dataclass(frozen=True)
class TensorDataRawConversionPlan:
    should_try_cpu_fallback: bool


class TensorDataCreatorResolutionError(TypeError):
    """Raised when a TensorData creator predicate returns a non-boolean result."""


class TensorDataCreatorNotFoundError(ValueError):
    """Raised when no TensorData creator matches the provided source."""


def normalize_backend_name(backend_name: str) -> str:
    if not isinstance(backend_name, str):
        raise TypeError(f'backend_name must be str, got {type(backend_name)}')

    normalized_name = backend_name.strip().lower()
    if normalized_name not in SUPPORTED_BACKEND_NAMES:
        supported = ', '.join(SUPPORTED_BACKEND_NAMES)
        raise ValueError(f'Unsupported backend_name: {backend_name}. Expected one of: {supported}')

    return normalized_name


def normalize_state(state: Union[StateEnum, str]) -> StateEnum:
    if isinstance(state, StateEnum):
        return state
    if isinstance(state, str):
        return StateEnum(state)
    raise TypeError(f'state must be StateEnum or str, got {type(state)}')


def normalize_task(task: Union[Task, str]) -> Task:
    if isinstance(task, Task):
        return task
    if isinstance(task, str):
        return Task(TaskTypesEnum(task))
    raise TypeError(f'task must be Task or str, got {type(task)}')


def normalize_optional_data_type(data_type: Optional[Union[DataTypesEnum, str]]) -> Optional[DataTypesEnum]:
    if data_type is None:
        return None
    if isinstance(data_type, str):
        mapped = LEGACY_DATA_TYPE_MAPPING.get(data_type)
        if mapped is not None:
            return mapped
        return LEGACY_DATA_TYPE_MAPPING.get(DataTypesEnum(data_type))
    if isinstance(data_type, DataTypesEnum):
        return LEGACY_DATA_TYPE_MAPPING[data_type]
    raise TypeError(f'data_type must be DataTypesEnum, str, or None, got {type(data_type)}')


def normalize_data_type(data_type: Union[DataTypesEnum, str]) -> DataTypesEnum:
    normalized_data_type = normalize_optional_data_type(data_type)
    if normalized_data_type is None:
        raise ValueError('data_type must not be None for TensorData runtime instances')
    return normalized_data_type


def normalize_tensordata_identity(
    task: Union[Task, str],
    data_type: Union[DataTypesEnum, str],
    state: Union[StateEnum, str],
) -> TensorDataIdentity:
    return TensorDataIdentity(
        task=normalize_task(task),
        data_type=normalize_data_type(data_type),
        state=normalize_state(state),
    )


def validate_creator_predicate_result(predicate: Callable[[Any], bool], result: Any) -> bool:
    if not isinstance(result, bool):
        raise TensorDataCreatorResolutionError(
            f'Predicate {predicate.__name__} must return bool, got {type(result)}'
        )
    return result


def build_backend_plan(backend_name: str) -> TensorDataBackendPlan:
    return TensorDataBackendPlan(backend_name=normalize_backend_name(backend_name))


def build_device_sync_plan(features_device_type: str, backend_device_type: str) -> TensorDataDeviceSyncPlan:
    return TensorDataDeviceSyncPlan(
        should_move_to_backend=features_device_type != backend_device_type
    )


def build_raw_conversion_plan(backend_name: str) -> TensorDataRawConversionPlan:
    normalized_backend_name = normalize_backend_name(backend_name)
    return TensorDataRawConversionPlan(
        should_try_cpu_fallback=normalized_backend_name != 'cpu'
    )


def resolve_registered_creator(
    creators: Iterable[tuple[Callable[[Any], bool], Callable]],
    source_data: Any,
) -> Callable:
    for predicate, creator in creators:
        result = validate_creator_predicate_result(
            predicate=predicate,
            result=predicate(source_data),
        )
        if result:
            return creator

    raise TensorDataCreatorNotFoundError(
        f'No creator registered for input: {type(source_data)}'
    )
