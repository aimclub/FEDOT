from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

from fedot.core.constants import AUTO_PRESET_NAME, BEST_QUALITY_PRESET_NAME
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


class RepositoryKind(Enum):
    all = 'all'
    model = 'model'
    data_operation = 'data_operation'
    automl = 'automl'


@dataclass(frozen=True)
class CatalogLoadError:
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OperationQuery:
    repository_kind: RepositoryKind = RepositoryKind.all
    task_type: Optional[TaskTypesEnum] = None
    data_type: Optional[DataTypesEnum] = None
    tags: Tuple[str, ...] = ()
    forbidden_tags: Tuple[str, ...] = ()
    preset: Optional[str] = None
    is_full_match: bool = False
    default_excluded_tags: Tuple[str, ...] = ()
    extra_ts_installed: bool = True


def parse_repository_kind(value: str) -> RepositoryKind:
    return RepositoryKind(value)


def normalize_operation_query(query: OperationQuery) -> OperationQuery:
    forbidden_tags = tuple(query.forbidden_tags)
    preset = normalize_preset_name(query.preset)

    if not query.tags:
        forbidden_tags = forbidden_tags + tuple(
            tag for tag in query.default_excluded_tags if tag not in forbidden_tags
        )

    if query.task_type is TaskTypesEnum.ts_forecasting and not query.extra_ts_installed and 'ts-extra' not in forbidden_tags:
        forbidden_tags = forbidden_tags + ('ts-extra',)

    return OperationQuery(
        repository_kind=query.repository_kind,
        task_type=query.task_type,
        data_type=query.data_type,
        tags=tuple(query.tags),
        forbidden_tags=forbidden_tags,
        preset=preset,
        is_full_match=query.is_full_match,
        default_excluded_tags=tuple(query.default_excluded_tags),
        extra_ts_installed=query.extra_ts_installed,
    )


def normalize_preset_name(preset: Optional[str]) -> Optional[str]:
    if preset is None:
        return None
    if BEST_QUALITY_PRESET_NAME in preset or AUTO_PRESET_NAME in preset:
        return None
    return preset


def filter_operation_infos(operations: Sequence[Any], query: OperationQuery) -> Tuple[Any, ...]:
    normalized_query = normalize_operation_query(query)
    return tuple(operation for operation in operations if matches_operation_query(operation, normalized_query))


def matches_operation_query(operation: Any, query: OperationQuery) -> bool:
    tags = tuple(getattr(operation, 'tags', ()) or ())
    presets = tuple(getattr(operation, 'presets', ()) or ())
    task_types = tuple(getattr(operation, 'task_type', ()) or ())
    input_types = tuple(getattr(operation, 'input_types', ()) or ())

    if query.task_type is not None and query.task_type not in task_types:
        return False

    if query.tags and not contains_tags(query.tags, tags, query.is_full_match):
        return False

    if query.forbidden_tags and contains_tags(query.forbidden_tags, tags, False):
        return False

    if not contains_preset(presets, query.preset):
        return False

    if query.data_type is None:
        return True

    if query.data_type in (DataTypesEnum.text, DataTypesEnum.image):
        return True

    valid_data_types = resolve_valid_data_types(query.data_type)
    return any(data_type in input_types for data_type in valid_data_types)


def contains_tags(candidate_tags: Iterable[str], operation_tags: Iterable[str], is_full_match: bool) -> bool:
    operation_tags = tuple(operation_tags or ())
    matches = tuple(tag in operation_tags for tag in candidate_tags)
    return all(matches) if is_full_match else any(matches)


def contains_preset(operation_presets: Iterable[str], preset: Optional[str]) -> bool:
    if preset is None:
        return True
    return preset in tuple(operation_presets or ())


def resolve_valid_data_types(data_type: DataTypesEnum) -> Tuple[DataTypesEnum, ...]:
    if data_type == DataTypesEnum.ts:
        return DataTypesEnum.ts, DataTypesEnum.table
    return (data_type,)
