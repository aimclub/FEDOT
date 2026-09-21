from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import RepositoryKind
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
    repository_kind: RepositoryKind = RepositoryKind.model


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
                tags=tuple(model.capabilities.tags),
                repository_kind=(RepositoryKind.model if isinstance(model, ExternalModelSpec)
                                 else RepositoryKind.data_operation),
            ))
    return tuple(views)


def filter_extension_operation_views(task_type: Optional[TaskTypesEnum],
                                     data_type: Optional[DataTypesEnum],
                                     tags: Optional[Sequence[str]] = None,
                                     forbidden_tags: Optional[Sequence[str]] = None,
                                     repository_kind: RepositoryKind = RepositoryKind.all,
                                     is_full_match: bool = False) -> Tuple[ExtensionOperationView, ...]:
    requested_tags = tuple(tags or ())
    forbidden = set(forbidden_tags or ())
    views = []
    for view in get_extension_operation_views():
        if repository_kind not in (RepositoryKind.all, view.repository_kind):
            continue
        if task_type is not None and task_type not in view.tasks:
            continue
        if data_type is not None:
            requested_type = build_extension_data_type_view((data_type,)).tensor_types[0]
            if requested_type not in build_extension_data_type_view(view.data_types).tensor_types:
                continue
        tag_matches = tuple(tag in view.tags for tag in requested_tags)
        if requested_tags and not (all(tag_matches) if is_full_match else any(tag_matches)):
            continue
        if forbidden and any(tag in forbidden for tag in view.tags):
            continue
        views.append(view)
    return tuple(views)


def get_extension_operation_names(task_type: Optional[TaskTypesEnum],
                                  data_type: Optional[DataTypesEnum],
                                  tags: Optional[Sequence[str]] = None,
                                  forbidden_tags: Optional[Sequence[str]] = None,
                                  repository_kind: RepositoryKind = RepositoryKind.all,
                                  is_full_match: bool = False) -> list[str]:
    return sorted(view.name for view in filter_extension_operation_views(
        task_type, data_type, tags, forbidden_tags, repository_kind, is_full_match))
