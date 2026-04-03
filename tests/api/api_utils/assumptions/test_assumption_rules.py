from types import SimpleNamespace

from fedot.api.api_utils.assumptions.assumption_rules import (
    build_operations_filter_decision,
    default_repository_name_for_data,
    exclude_operations,
    finalize_operations,
    merge_preset_operations,
    parse_preset_spec,
    required_operations_for_data,
    resolve_explicit_suitable_operations,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


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


def test_required_operations_for_multimodal_image_include_sources_and_cnn():
    required = required_operations_for_data(_FakeMultiModalData(DataTypesEnum.image), DataTypesEnum.image)
    assert required == ('data_source_img', 'cnn')


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


def test_build_operations_filter_decision_keeps_required_image_operations():
    decision = build_operations_filter_decision(
        data=_FakeMultiModalData(DataTypesEnum.image),
        data_type=DataTypesEnum.image,
        available_operations=['rf'],
        suitable_operations=['rf'],
    )

    assert decision.allow_filtering is True
    assert decision.sampling_choices == ('rf', 'data_source_img', 'cnn')


def test_resolve_explicit_suitable_operations_keeps_non_default_if_explicitly_requested():
    operations_metadata = [
        SimpleNamespace(
            id='industrial_inception_nn',
            tags=('non-default', 'gpu_bridge'),
            task_type=(TaskTypesEnum.classification,),
            input_types=(DataTypesEnum.table,),
            presets=(),
        ),
        SimpleNamespace(
            id='rf',
            tags=('tree',),
            task_type=(TaskTypesEnum.classification,),
            input_types=(DataTypesEnum.table,),
            presets=(),
        ),
        SimpleNamespace(
            id='ridge',
            tags=('linear',),
            task_type=(TaskTypesEnum.regression,),
            input_types=(DataTypesEnum.table,),
            presets=(),
        ),
    ]

    suitable = resolve_explicit_suitable_operations(
        repository_kind='model',
        operations_metadata=operations_metadata,
        task_type=TaskTypesEnum.classification,
        data_type=DataTypesEnum.table,
        available_operations=['industrial_inception_nn', 'ridge'],
    )

    assert suitable == ('industrial_inception_nn',)


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