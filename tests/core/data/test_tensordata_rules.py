import pytest

from fedot.core.data.tensordata_rules import (
    TensorDataCreatorNotFoundError,
    TensorDataCreatorResolutionError,
    build_backend_plan,
    build_device_sync_plan,
    build_raw_conversion_plan,
    normalize_backend_name,
    normalize_optional_data_type,
    normalize_tensordata_identity,
    resolve_registered_creator,
    validate_creator_predicate_result,
)
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


@pytest.mark.unit
@pytest.mark.parametrize(
    'backend_name, expected',
    [
        ('cpu', 'cpu'),
        ('CPU', 'cpu'),
        (' gpu ', 'gpu'),
    ],
)
def test_normalize_backend_name_accepts_supported_values(backend_name, expected):
    assert normalize_backend_name(backend_name) == expected


@pytest.mark.unit
def test_normalize_backend_name_rejects_unknown_value():
    with pytest.raises(ValueError, match='Unsupported backend_name'):
        normalize_backend_name('tpu')


@pytest.mark.unit
@pytest.mark.parametrize(
    'raw_data_type, expected',
    [
        (DataTypesEnum.table, DataTypesEnum.tabular),
        (DataTypesEnum.text, DataTypesEnum.tabular),
        ('table', DataTypesEnum.tabular),
        ('text', DataTypesEnum.tabular),
        (DataTypesEnum.multi_ts, DataTypesEnum.ts),
        (DataTypesEnum.image, DataTypesEnum.ts),
        ('multi_time_series', DataTypesEnum.ts),
        ('image', DataTypesEnum.ts),
    ],
)
def test_normalize_optional_data_type_maps_legacy_values(raw_data_type, expected):
    assert normalize_optional_data_type(raw_data_type) == expected


@pytest.mark.unit
def test_normalize_optional_data_type_is_idempotent_for_canonical_values():
    assert normalize_optional_data_type(DataTypesEnum.tabular) == DataTypesEnum.tabular
    assert normalize_optional_data_type(DataTypesEnum.ts) == DataTypesEnum.ts


@pytest.mark.unit
def test_normalize_tensordata_identity_converts_strings_to_typed_values():
    identity = normalize_tensordata_identity(
        task='classification',
        data_type='image',
        state='predict',
    )

    assert identity.task.task_type == TaskTypesEnum.classification
    assert identity.data_type == DataTypesEnum.ts
    assert identity.state == StateEnum.PREDICT


@pytest.mark.unit
def test_validate_creator_predicate_result_rejects_non_boolean_result():
    def bad_predicate(_):
        return 'yes'

    with pytest.raises(TensorDataCreatorResolutionError, match='must return bool'):
        validate_creator_predicate_result(bad_predicate, bad_predicate(None))


@pytest.mark.unit
def test_build_backend_plan_returns_normalized_backend_name():
    plan = build_backend_plan(' GPU ')

    assert plan.backend_name == 'gpu'


@pytest.mark.unit
def test_build_raw_conversion_plan_enables_cpu_fallback_only_for_non_cpu_backend():
    assert build_raw_conversion_plan('gpu').should_try_cpu_fallback is True
    assert build_raw_conversion_plan('cpu').should_try_cpu_fallback is False


@pytest.mark.unit
def test_build_device_sync_plan_detects_when_move_is_needed():
    assert build_device_sync_plan('cpu', 'cuda').should_move_to_backend is True
    assert build_device_sync_plan('cpu', 'cpu').should_move_to_backend is False


@pytest.mark.unit
def test_resolve_registered_creator_returns_first_matching_creator():
    creator_a = object()
    creator_b = object()

    creators = [
        (lambda _: False, creator_a),
        (lambda _: True, creator_b),
    ]

    assert resolve_registered_creator(creators, source_data={'x': 1}) is creator_b


@pytest.mark.unit
def test_resolve_registered_creator_rejects_missing_creator():
    creators = [(lambda _: False, object())]

    with pytest.raises(TensorDataCreatorNotFoundError, match='No creator registered'):
        resolve_registered_creator(creators, source_data={'x': 1})
