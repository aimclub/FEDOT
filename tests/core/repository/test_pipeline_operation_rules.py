from fedot.core.repository.pipeline_operation_rules import (
    build_pipeline_operations_by_role,
    filter_available_pipeline_operations,
)
from fedot.core.repository.tasks import TaskTypesEnum


def test_filter_available_pipeline_operations_returns_sorted_intersection():
    filtered = filter_available_pipeline_operations(
        preset_operations=['ridge', 'rf', 'external_linear'],
        available_operations=['external_linear', 'ridge', 'external_linear'],
    )

    assert filtered == ('external_linear', 'ridge')


def test_build_pipeline_operations_by_role_returns_all_operations_for_non_ts_task():
    operations_by_role = build_pipeline_operations_by_role(
        available_operations=['ridge', 'external_linear'],
        task_type=TaskTypesEnum.regression,
    )

    assert operations_by_role.primary == ('external_linear', 'ridge')
    assert operations_by_role.secondary == ('external_linear', 'ridge')


def test_build_pipeline_operations_by_role_uses_non_lagged_ts_subset_for_primary_nodes():
    operations_by_role = build_pipeline_operations_by_role(
        available_operations=['external_non_lagged', 'lagged', 'ridge'],
        task_type=TaskTypesEnum.ts_forecasting,
        ts_data_operations=['exog_ts', 'lagged'],
        ts_primary_models=['external_non_lagged'],
    )

    assert operations_by_role.primary == ('external_non_lagged', 'lagged')
    assert operations_by_role.secondary == (
        'external_non_lagged', 'lagged', 'ridge')
