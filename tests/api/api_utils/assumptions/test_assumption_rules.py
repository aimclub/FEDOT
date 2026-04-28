from fedot.api.api_utils.assumptions.assumption_rules import (
    build_operations_filter_decision,
    default_repository_name_for_data,
    exclude_operations,
    finalize_operations,
    merge_preset_operations,
    parse_preset_spec,
    required_operations_for_data,
)
from fedot.core.repository.dataset_types import DataTypesEnum


class _FakeData:
    def __init__(self, data_type):
        self.data_type = data_type


class _FakeMultiModalData(dict):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type


def test_default_repository_name_for_multi_ts():
    assert default_repository_name_for_data(_FakeData(DataTypesEnum.multi_ts)) == 'all'


def test_default_repository_name_for_regular_data():
    assert default_repository_name_for_data(_FakeData(DataTypesEnum.table)) == 'model'


def test_required_operations_for_multimodal_ts_tensor_include_time_series_source():
    required = required_operations_for_data(_FakeMultiModalData(DataTypesEnum.ts), DataTypesEnum.ts)
    assert required == ('data_source_time_series',)


def test_required_operations_normalize_legacy_image_to_ts():
    required = required_operations_for_data(_FakeMultiModalData(DataTypesEnum.image), DataTypesEnum.image)
    assert required == ('data_source_time_series',)


def test_build_operations_filter_decision_intersects_with_suitable_operations():
    decision = build_operations_filter_decision(
        data=_FakeData(DataTypesEnum.table),
        data_type=DataTypesEnum.table,
        available_operations=['rf', 'lasso', 'xgboost'],
        suitable_operations=['rf', 'xgboost', 'ridge'],
    )

    assert decision.allow_filtering is True
    assert decision.whitelist == ('rf', 'lasso', 'xgboost')
    assert decision.sampling_choices == ('rf', 'xgboost')


def test_build_operations_filter_decision_keeps_required_ts_tensor_operations():
    decision = build_operations_filter_decision(
        data=_FakeMultiModalData(DataTypesEnum.ts),
        data_type=DataTypesEnum.ts,
        available_operations=['rf'],
        suitable_operations=['rf'],
    )

    assert decision.allow_filtering is True
    assert decision.sampling_choices == ('rf', 'data_source_time_series')


def test_parse_preset_spec_extracts_stable_and_modification_flags():
    spec = parse_preset_spec('stable*tree')

    assert spec.base_preset == 'best_quality'
    assert spec.modification == '*tree'
    assert spec.use_stable is True
    assert spec.use_auto is False
    assert spec.use_gpu is False


def test_merge_exclude_and_finalize_operations_are_deterministic():
    merged = merge_preset_operations(['rf', 'xgboost', 'knn'], ['xgboost', 'rf'])
    filtered = exclude_operations(merged, ['xgboost'])

    assert merged == ('rf', 'xgboost')
    assert filtered == ('rf',)
    assert finalize_operations(['rf', 'rf', 'knn']) == ['knn', 'rf']
