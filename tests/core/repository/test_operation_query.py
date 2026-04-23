from dataclasses import dataclass

from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import (
    OperationQuery,
    RepositoryKind,
    contains_tags,
    filter_operation_infos,
    normalize_operation_query,
    normalize_preset_name,
)
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class FakeOperation:
    id: str
    task_type: tuple
    input_types: tuple
    tags: tuple = ()
    presets: tuple = ()


BASE_OPERATION = FakeOperation(
    id='rf',
    task_type=(TaskTypesEnum.classification, TaskTypesEnum.regression),
    input_types=(DataTypesEnum.table,),
    tags=('tree', 'simple'),
    presets=('fast_train',),
)

TS_OPERATION = FakeOperation(
    id='lagged',
    task_type=(TaskTypesEnum.ts_forecasting,),
    input_types=(DataTypesEnum.ts, DataTypesEnum.table),
    tags=('ts-extra', 'lagged'),
    presets=('ts',),
)


def test_contains_tags_supports_partial_and_full_match():
    assert contains_tags(('tree',), ('tree', 'simple'), False) is True
    assert contains_tags(('tree', 'simple'), ('tree', 'simple'), True) is True
    assert contains_tags(('tree', 'missing'), ('tree', 'simple'), True) is False


def test_normalize_preset_name_resets_auto_like_presets():
    assert normalize_preset_name(None) is None
    assert normalize_preset_name(f'{AUTO_PRESET_NAME}*tree') is None
    assert normalize_preset_name('fast_train') == 'fast_train'


def test_normalize_operation_query_applies_default_excluded_tags_only_when_tags_missing():
    query = OperationQuery(
        repository_kind=RepositoryKind.model,
        default_excluded_tags=('deprecated', 'expensive'),
    )

    normalized = normalize_operation_query(query)

    assert normalized.forbidden_tags == ('deprecated', 'expensive')


def test_filter_operation_infos_respects_tags_and_presets():
    query = OperationQuery(
        repository_kind=RepositoryKind.model,
        task_type=TaskTypesEnum.classification,
        data_type=DataTypesEnum.table,
        tags=('tree',),
        preset='fast_train',
    )

    filtered = filter_operation_infos((BASE_OPERATION, TS_OPERATION), query)

    assert filtered == (BASE_OPERATION,)


def test_filter_operation_infos_excludes_ts_extra_when_optional_dependency_missing():
    query = OperationQuery(
        repository_kind=RepositoryKind.model,
        task_type=TaskTypesEnum.ts_forecasting,
        data_type=DataTypesEnum.ts,
        extra_ts_installed=False,
    )

    filtered = filter_operation_infos((BASE_OPERATION, TS_OPERATION), query)

    assert filtered == ()
