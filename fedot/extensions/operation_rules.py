from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import RepositoryKind
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.registry import get_registered_extensions


@dataclass(frozen=True)
class ExtensionOperationView:
    name: str
    tasks: Tuple[TaskTypesEnum, ...]
    data_types: Tuple[DataTypesEnum, ...]
    tags: Tuple[str, ...]


def should_include_extensions(repository_kind: RepositoryKind) -> bool:
    return repository_kind in (RepositoryKind.model, RepositoryKind.all)


def get_extension_operation_views() -> Tuple[ExtensionOperationView, ...]:
    views = []
    for registered_extension in get_registered_extensions():
        for model in registered_extension.manifest.models:
            views.append(ExtensionOperationView(
                name=model.name,
                tasks=tuple(model.capabilities.tasks),
                data_types=tuple(model.capabilities.data_types),
                tags=tuple(model.capabilities.tags),
            ))
    return tuple(views)


def filter_extension_operation_views(task_type: Optional[TaskTypesEnum],
                                     data_type: Optional[DataTypesEnum],
                                     tags: Optional[Sequence[str]] = None,
                                     forbidden_tags: Optional[Sequence[str]] = None) -> Tuple[ExtensionOperationView, ...]:
    requested_tags = tuple(tags or ())
    forbidden = set(forbidden_tags or ())
    views = []
    for view in get_extension_operation_views():
        if task_type is not None and task_type not in view.tasks:
            continue
        if data_type is not None and data_type not in view.data_types:
            continue
        if requested_tags and not any(tag in view.tags for tag in requested_tags):
            continue
        if forbidden and any(tag in forbidden for tag in view.tags):
            continue
        views.append(view)
    return tuple(views)


def get_extension_operation_names(task_type: Optional[TaskTypesEnum],
                                  data_type: Optional[DataTypesEnum],
                                  tags: Optional[Sequence[str]] = None,
                                  forbidden_tags: Optional[Sequence[str]] = None) -> list[str]:
    return sorted(view.name for view in filter_extension_operation_views(task_type, data_type, tags, forbidden_tags))
