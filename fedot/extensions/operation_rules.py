from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import (
    OperationQuery,
    RepositoryKind,
    matches_operation_query,
    normalize_operation_query,
)
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.registry import get_registered_extensions
from fedot.extensions.contracts import ExternalModelSpec
from fedot.extensions.data_type_rules import build_extension_data_type_view


@dataclass(frozen=True)
class ExtensionOperationView:
    name: str
    tasks: Tuple[TaskTypesEnum, ...]
    data_types: Tuple[DataTypesEnum, ...]
    tags: Tuple[str, ...]
    presets: Tuple[str, ...] = ()
    repository_kind: RepositoryKind = RepositoryKind.model

    @property
    def task_type(self) -> Tuple[TaskTypesEnum, ...]:
        return self.tasks

    @property
    def input_types(self) -> Tuple[DataTypesEnum, ...]:
        return build_extension_data_type_view(self.data_types).input_types


def should_include_extensions(repository_kind: RepositoryKind) -> bool:
    return repository_kind in (RepositoryKind.model, RepositoryKind.data_operation, RepositoryKind.all)


def get_extension_operation_views() -> Tuple[ExtensionOperationView, ...]:
    views = []
    for registered_extension in get_registered_extensions():
        manifest = registered_extension.manifest
        for model in manifest.models + manifest.transforms:
            views.append(ExtensionOperationView(
                name=model.name,
                tasks=tuple(model.capabilities.tasks),
                data_types=tuple(model.capabilities.data_types),
                tags=tuple(dict.fromkeys(
                    ('external',) + model.capabilities.tags)),
                repository_kind=(RepositoryKind.model if isinstance(model, ExternalModelSpec)
                                 else RepositoryKind.data_operation),
            ))
    return tuple(views)


def _operation_query(task_type,
                     data_type,
                     tags,
                     forbidden_tags,
                     repository_kind,
                     is_full_match) -> OperationQuery:
    if isinstance(task_type, OperationQuery):
        return task_type
    return OperationQuery(
        repository_kind=repository_kind,
        task_type=task_type,
        data_type=data_type,
        tags=tuple(tags or ()),
        forbidden_tags=tuple(forbidden_tags or ()),
        is_full_match=is_full_match,
    )


def filter_extension_operation_views(task_type: Optional[TaskTypesEnum] = None,
                                     data_type: Optional[DataTypesEnum] = None,
                                     tags: Optional[Sequence[str]] = None,
                                     forbidden_tags: Optional[Sequence[str]] = None,
                                     repository_kind: RepositoryKind = RepositoryKind.all,
                                     is_full_match: bool = False) -> Tuple[ExtensionOperationView, ...]:
    query = _operation_query(
        task_type, data_type, tags, forbidden_tags, repository_kind, is_full_match)
    normalized_query = normalize_operation_query(query)
    return tuple(
        view
        for view in get_extension_operation_views()
        if normalized_query.repository_kind in (RepositoryKind.all, view.repository_kind)
        and matches_operation_query(view, normalized_query)
    )


def get_extension_operation_names(task_type: Optional[TaskTypesEnum] = None,
                                  data_type: Optional[DataTypesEnum] = None,
                                  tags: Optional[Sequence[str]] = None,
                                  forbidden_tags: Optional[Sequence[str]] = None,
                                  repository_kind: RepositoryKind = RepositoryKind.all,
                                  is_full_match: bool = False) -> list[str]:
    return sorted(view.name for view in filter_extension_operation_views(
        task_type, data_type, tags, forbidden_tags, repository_kind, is_full_match))
