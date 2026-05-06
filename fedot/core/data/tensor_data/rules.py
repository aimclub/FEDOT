from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union, List

from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

SUPPORTED_BACKEND_NAMES = ('cpu', 'gpu')

DEFAULT_DATALOADER_KWARGS = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 0,
    'drop_last': False,
}

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
class LoadDataSpecNormalization:
    task: Task
    data_type: Optional[DataTypesEnum]
    state: StateEnum
    ts_orientation: Optional[TSOrientationEnum]
    embedding_strategy: dict
    dataloader_kwargs: dict


@dataclass(frozen=True)
class TensorDataCreationRequest:
    backend_name: str


@dataclass(frozen=True)
class TensorDataCreationFailure:
    source_type_name: str
    backend_name: str
    error_type_name: str

    @property
    def message(self) -> str:
        return (
            f"Error creating TensorData for source_type={self.source_type_name} "
            f"on backend={self.backend_name}: {self.error_type_name}"
        )


@dataclass(frozen=True)
class TensorDataTabularFileLoadPlan:
    file_path: str
    delimiter: str
    possible_idx_keywords: list[str]


@dataclass(frozen=True)
class TensorDataDeviceSyncPlan:
    should_move_to_backend: bool


@dataclass
class TensorDataRawConversionPlan:
    should_try_cpu_fallback: bool
    preprocessing_done: bool


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


def normalize_optional_ts_orientation(
    ts_orientation: Optional[Union[TSOrientationEnum, str]],
) -> Optional[TSOrientationEnum]:
    if ts_orientation is None:
        return None
    if isinstance(ts_orientation, TSOrientationEnum):
        return ts_orientation
    if isinstance(ts_orientation, str):
        return TSOrientationEnum(ts_orientation)
    raise TypeError(
        f'ts_orientation must be TSOrientationEnum, str, or None, got {type(ts_orientation)}'
    )


def normalize_embedding_strategy(embedding_strategy: Optional[List]) -> dict:
    if embedding_strategy is None:
        return embedding_strategy
    elif isinstance(embedding_strategy, dict):
        return [embedding_strategy]
    elif isinstance(embedding_strategy, list):
        return embedding_strategy
    else:
        raise TypeError(
            f'embedding_strategy must be dict or None, got {type(embedding_strategy)}'
        )


def normalize_dataloader_kwargs(dataloader_kwargs: Optional[dict]) -> dict:
    if dataloader_kwargs is None:
        return dict(DEFAULT_DATALOADER_KWARGS)
    if not isinstance(dataloader_kwargs, dict):
        raise TypeError(
            f'dataloader_kwargs must be dict or None, got {type(dataloader_kwargs)}'
        )
    return {**DEFAULT_DATALOADER_KWARGS, **dataloader_kwargs}


def build_load_data_spec_normalization(
    task: Union[Task, str],
    data_type: Optional[Union[DataTypesEnum, str]],
    state: Union[StateEnum, str],
    ts_orientation: Optional[Union[TSOrientationEnum, str]],
    embedding_strategy: Optional[dict],
    dataloader_kwargs: Optional[dict],
) -> LoadDataSpecNormalization:
    return LoadDataSpecNormalization(
        task=normalize_task(task),
        data_type=normalize_optional_data_type(data_type),
        state=normalize_state(state),
        ts_orientation=normalize_optional_ts_orientation(ts_orientation),
        embedding_strategy=normalize_embedding_strategy(embedding_strategy),
        dataloader_kwargs=normalize_dataloader_kwargs(dataloader_kwargs),
    )


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


def build_creation_request(backend_name: str) -> TensorDataCreationRequest:
    return TensorDataCreationRequest(backend_name=normalize_backend_name(backend_name))


def build_creation_failure(
    source_data: Any,
    backend_name: str,
    error: Exception,
) -> TensorDataCreationFailure:
    return TensorDataCreationFailure(
        source_type_name=type(source_data).__name__,
        backend_name=normalize_backend_name(backend_name),
        error_type_name=type(error).__name__,
    )


def normalize_array_target_reference(
    target: Any,
    target_idx: Any,
    feature_width: int,
) -> tuple[Any, Any]:
    if target_idx is not None:
        return target, target_idx

    if isinstance(target, int) and 0 <= target < feature_width:
        return None, target

    return target, target_idx


def normalize_tabular_file_delimiter(file_path: str, delimiter: str) -> str:
    if file_path.lower().endswith('.tsv') and delimiter == ',':
        return '\t'
    return delimiter


def normalize_possible_idx_keywords(
    possible_idx_keywords: Optional[list[str]],
    default_keywords: list[str],
) -> list[str]:
    if possible_idx_keywords is None:
        return list(default_keywords)
    return list(possible_idx_keywords)


def validate_tabular_file_path(file_path: str) -> str:
    normalized_path = str(file_path)
    supported_suffixes = ('.csv', '.tsv')

    if not normalized_path.lower().endswith(supported_suffixes):
        raise ValueError(f'Unsupported tabular file format: {normalized_path}')

    if not Path(normalized_path).is_file():
        raise ValueError(f'File {normalized_path} does not exist')

    return normalized_path


def build_tabular_file_load_plan(
    file_path: str,
    delimiter: str,
    possible_idx_keywords: Optional[list[str]],
    default_keywords: list[str],
) -> TensorDataTabularFileLoadPlan:
    normalized_path = validate_tabular_file_path(file_path)
    return TensorDataTabularFileLoadPlan(
        file_path=normalized_path,
        delimiter=normalize_tabular_file_delimiter(normalized_path, delimiter),
        possible_idx_keywords=normalize_possible_idx_keywords(
            possible_idx_keywords=possible_idx_keywords,
            default_keywords=default_keywords,
        ),
    )


def build_device_sync_plan(features_device_type: str, backend_device_type: str) -> TensorDataDeviceSyncPlan:
    return TensorDataDeviceSyncPlan(
        should_move_to_backend=features_device_type != backend_device_type
    )


def build_raw_conversion_plan(backend_name: str) -> TensorDataRawConversionPlan:
    normalized_backend_name = normalize_backend_name(backend_name)
    return TensorDataRawConversionPlan(
        should_try_cpu_fallback=normalized_backend_name != 'cpu',
        preprocessing_done=False
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
