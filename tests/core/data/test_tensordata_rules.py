import pytest

from fedot.core.data.tensor_data.rules import (
    TensorDataCreatorNotFoundError,
    DEFAULT_DATALOADER_KWARGS,
    TensorDataCreatorResolutionError,
    build_backend_plan,
    build_creation_failure,
    build_creation_request,
    build_device_sync_plan,
    build_load_data_spec_normalization,
    build_raw_conversion_plan,
    build_tabular_file_load_plan,
    normalize_array_target_reference,
    normalize_backend_name,
    normalize_dataloader_kwargs,
    normalize_optional_ts_orientation,
    normalize_possible_idx_keywords,
    normalize_tabular_file_delimiter,
    normalize_optional_data_type,
    normalize_tensordata_identity,
    resolve_registered_creator,
    validate_creator_predicate_result,
)
from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
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


@pytest.mark.unit
def test_build_creation_request_normalizes_backend_name():
    request = build_creation_request(' GPU ')

    assert request.backend_name == 'gpu'


@pytest.mark.unit
def test_build_creation_failure_keeps_source_type_and_error_type():
    failure = build_creation_failure({'x': 1}, 'cpu', RuntimeError('boom'))

    assert failure.source_type_name == 'dict'
    assert failure.backend_name == 'cpu'
    assert failure.error_type_name == 'RuntimeError'
    assert 'source_type=dict' in failure.message


@pytest.mark.unit
def test_normalize_array_target_reference_moves_integer_target_to_target_idx():
    target, target_idx = normalize_array_target_reference(2, None, feature_width=5)

    assert target is None
    assert target_idx == 2


@pytest.mark.unit
def test_normalize_array_target_reference_keeps_explicit_target_idx_priority():
    target, target_idx = normalize_array_target_reference(2, 1, feature_width=5)

    assert target == 2
    assert target_idx == 1


@pytest.mark.unit
def test_normalize_tabular_file_delimiter_infers_tsv_separator_from_suffix():
    assert normalize_tabular_file_delimiter('sample.tsv', ',') == '\t'
    assert normalize_tabular_file_delimiter('sample.csv', ',') == ','


@pytest.mark.unit
def test_normalize_possible_idx_keywords_uses_defaults_when_not_provided():
    assert normalize_possible_idx_keywords(None, ['idx', 'id']) == ['idx', 'id']
    assert normalize_possible_idx_keywords(['custom'], ['idx']) == ['custom']


@pytest.mark.unit
def test_build_tabular_file_load_plan_normalizes_defaults_for_tsv(tmp_path):
    file_path = tmp_path / 'data.tsv'
    file_path.write_text('idx\tvalue\n1\t2\n')

    plan = build_tabular_file_load_plan(
        file_path=str(file_path),
        delimiter=',',
        possible_idx_keywords=None,
        default_keywords=['idx', 'id'],
    )

    assert plan.file_path == str(file_path)
    assert plan.delimiter == '\t'
    assert plan.possible_idx_keywords == ['idx', 'id']


@pytest.mark.unit
def test_build_tabular_file_load_plan_rejects_missing_file(tmp_path):
    missing_path = tmp_path / 'missing.csv'

    with pytest.raises(ValueError, match='does not exist'):
        build_tabular_file_load_plan(
            file_path=str(missing_path),
            delimiter=',',
            possible_idx_keywords=None,
            default_keywords=['idx'],
        )


@pytest.mark.unit
def test_normalize_optional_ts_orientation_converts_string_to_enum():
    assert normalize_optional_ts_orientation('long') == TSOrientationEnum.long
    assert normalize_optional_ts_orientation(TSOrientationEnum.wide) == TSOrientationEnum.wide


@pytest.mark.unit
def test_normalize_dataloader_kwargs_merges_defaults_without_mutating_input():
    user_kwargs = {'batch_size': 64}

    normalized_kwargs = normalize_dataloader_kwargs(user_kwargs)

    assert normalized_kwargs == {**DEFAULT_DATALOADER_KWARGS, 'batch_size': 64}
    assert user_kwargs == {'batch_size': 64}


@pytest.mark.unit
def test_build_load_data_spec_normalization_is_idempotent_for_normalized_values():
    first = build_load_data_spec_normalization(
        task='classification',
        data_type='table',
        state='predict',
        ts_orientation='long',
        embedding_strategy={'model_name': 'demo'},
        dataloader_kwargs={'batch_size': 64},
    )

    second = build_load_data_spec_normalization(
        task=first.task,
        data_type=first.data_type,
        state=first.state,
        ts_orientation=first.ts_orientation,
        embedding_strategy=first.embedding_strategy,
        dataloader_kwargs=first.dataloader_kwargs,
    )

    assert second == first
    assert second.embedding_strategy is not first.embedding_strategy
    assert second.dataloader_kwargs is not first.dataloader_kwargs


@pytest.mark.unit
def test_build_load_data_spec_normalization_keeps_empty_embedding_strategy_deterministic():
    normalization = build_load_data_spec_normalization(
        task='classification',
        data_type='table',
        state='fit',
        ts_orientation=None,
        embedding_strategy=None,
        dataloader_kwargs=None,
    )

    assert normalization.embedding_strategy == {}
    assert normalization.dataloader_kwargs == DEFAULT_DATALOADER_KWARGS
