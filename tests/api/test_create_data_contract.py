"""Public-boundary regressions for FED-01 (independent properties follow later)."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.api.create_data import create_data_lazy
from fedot.core.backend.backend import Backend
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.contracts import TensorDataContractError
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.preprocessing.methods.categorical_encoding import LabelEncoder
from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum


pytestmark = pytest.mark.unit


def test_specs_do_not_share_task_or_nested_options():
    strategy = [{'method': EncodingMethodEnum.label, 'features_idx': [0]}]
    first = DataSpec(encoding_strategy=strategy, categorical_idx=[0])
    second = DataSpec()
    assert first.task is not second.task
    first.task.task_params = {'changed': True}
    assert second.task.task_params != first.task.task_params
    first.encoding_strategy[0]['features_idx'].append(1)
    assert strategy[0]['features_idx'] == [0]


@pytest.mark.parametrize('kind', ['numpy', 'torch', 'dataframe'])
def test_creation_owns_arrays_and_filters_sample_labels(kind):
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source = torch.from_numpy(
        values.copy()) if kind == 'torch' else values.copy()
    if kind == 'dataframe':
        source = pd.DataFrame(source, columns=['a', 'b', 'c'])
    target = np.array([0, np.nan, 1, 0])
    labels = np.array(['row-d', 'row-a', 'row-c', 'row-b'])
    result = create_data(source, target=target, idx=labels, use_cache=False)
    np.testing.assert_array_equal(result.idx, ['row-d', 'row-c', 'row-b'])
    torch.testing.assert_close(
        result.features, torch.from_numpy(values[[0, 2, 3]]))
    assert result.idx_mapping == {0: 0, 1: 1, 2: 2}
    result.features[0, 0] = -100
    actual = source.to_numpy() if kind == 'dataframe' else np.asarray(source)
    np.testing.assert_array_equal(actual, values)
    assert np.isnan(target[1])
    np.testing.assert_array_equal(labels, ['row-d', 'row-a', 'row-c', 'row-b'])


@pytest.mark.parametrize('position', [0, 1, -1])
def test_target_extraction_preserves_column_ownership_and_feature_names(position):
    frame = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 1, 0], 'c': [8, 9, 7]})
    pos = position % 3
    result = create_data(frame, target_idx=position, use_cache=False)
    kept = [i for i in range(3) if i != pos]
    assert result.idx_mapping == dict(enumerate(kept))
    assert result.features_names == [frame.columns[i] for i in kept]
    assert len(result.idx) == 3
    np.testing.assert_array_equal(result.target[:, 0], frame.iloc[:, pos])


def test_string_target_does_not_replace_features_and_from_data_never_fits(monkeypatch):
    frame = pd.DataFrame({'a': [10., 20., 30.], 'label': [
                         'yes', 'no', 'yes'], 'color': ['b', 'a', 'b']})
    original = frame.copy(deep=True)
    train = create_data(frame, target='label', use_cache=False)
    assert train.features.shape == (3, 2)
    np.testing.assert_array_equal(train.features[:, 0], [10, 20, 30])
    assert train.idx_mapping == {0: 0, 1: 2}
    assert train.categorical_idx == [1]
    assert train.features_names == ['a', 'color']
    assert train.target.shape == (3, 1)
    pd.testing.assert_frame_equal(frame, original)
    before = deepcopy(train)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError('predict tried to fit a handler')

    monkeypatch.setattr(LabelEncoder, 'fit', forbidden_fit)
    incoming = pd.DataFrame({'a': [40., 50.], 'color': ['new', 'a']})
    for _ in range(2):
        predicted = create_data(incoming, from_data=train, use_cache=False)
        np.testing.assert_array_equal(predicted.features, [[40, 2], [50, 0]])
        assert predicted.target is None
        assert predicted.idx_mapping == {0: 0, 1: 2}
        assert train == before
        assert train.preparation_state.steps == before.preparation_state.steps


def test_from_data_reuses_in_memory_state_without_trace_files(isolated_cache_dir, monkeypatch):
    train = create_data(np.array(
        [['a'], ['b']], dtype=object), target=np.array([0, 1]), use_cache=False)
    from fedot.core.caching.tracer import TraceBuilder

    def forbidden_trace(*args, **kwargs):
        raise AssertionError('from_data must not reload a trace')

    monkeypatch.setattr(TraceBuilder, 'from_trace_uuid', forbidden_trace)
    predicted = create_data(
        np.array([['b']], dtype=object), from_data=train, use_cache=False)
    assert predicted.features.item() == 1


@pytest.mark.parametrize('changed', [
    pd.DataFrame({'b': [1.], 'a': [2.]}),
    pd.DataFrame({'a': [1.], 'c': [2.]}),
    np.ones((2, 3)),
])
def test_prediction_rejects_schema_drift(changed):
    train = create_data(pd.DataFrame(
        {'a': [1., 2.], 'b': [3., 4.]}), target=np.array([0, 1]))
    with pytest.raises(TensorDataContractError) as error:
        create_data(changed, from_data=train)
    assert error.value.code == 'schema_mismatch'


@pytest.mark.parametrize('options,field', [
    ({'state': 'fit'}, 'state'),
    ({'task': 'regression'}, 'task'),
    ({'encoding_strategy': []}, 'encoding_strategy'),
    ({'trace_uuid': 'unrelated'}, 'trace_uuid'),
])
def test_from_data_rejects_conflicting_preparation(options, field):
    train = create_data(np.ones((3, 2)), target=np.array(
        [0, 1, 0]), use_cache=False)
    with pytest.raises(TensorDataContractError) as error:
        create_data(np.ones((1, 2)), from_data=train, **options)
    assert error.value.field == field


@pytest.mark.parametrize('options,code', [
    ({'target_idx': [0, -2]}, 'duplicate_selector'),
    ({'target_idx': [True, False]}, 'invalid_selector'),
    ({'target_idx': [2]}, 'index_bounds'),
    ({'target_idx': ['a', 0]}, 'invalid_selector'),
    ({'idx': [1]}, 'row_alignment'),
    ({'target': np.ones(4)}, 'row_alignment'),
])
def test_invalid_input_has_stable_contract_error(options, code):
    with pytest.raises(TensorDataContractError) as error:
        create_data(np.ones((3, 2)), **options)
    assert error.value.code == code


@pytest.mark.parametrize('shape', [(), (0, 2), (2, 0), (2, 3, 4)])
def test_tabular_rejects_invalid_axes(shape):
    with pytest.raises(TensorDataContractError, match='features'):
        create_data(np.ones(shape), without_target=True)


@pytest.mark.parametrize('shape', [(3, 8), (3, 2, 8)])
def test_forecast_horizon_splits_time_not_samples(shape):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    train = create_data(source, data_type='time_series',
                        ts_forecast_horizon=2, use_cache=False)
    np.testing.assert_array_equal(train.features, source[..., :-2])
    np.testing.assert_array_equal(train.target, source[..., -2:])
    assert len(train.idx) == shape[0]
    assert train.ts_init_shape == source[..., :-2].shape
    test = create_data(source[..., :5], from_data=train, use_cache=False)
    np.testing.assert_array_equal(test.features, source[..., :5])
    assert test.target is None
    assert len(test.idx_mapping) == test.features.shape[1]


@pytest.mark.parametrize('horizon', [0, -1, True, 1.5, 5, 6])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(TensorDataContractError) as error:
        create_data(np.ones((2, 5)), data_type='time_series',
                    ts_forecast_horizon=horizon)
    assert error.value.code == 'invalid_horizon'


def test_one_dimensional_sources_have_unambiguous_axes():
    source = np.arange(6, dtype=np.float32)
    tabular = create_data(source, without_target=True)
    series = create_data(source, data_type='time_series')
    assert tabular.features.shape == (6, 1)
    assert len(tabular.idx) == 6
    assert series.features.shape == (1, 6)
    assert len(series.idx) == 1


def test_single_series_explicit_horizon_vector_is_not_overwritten():
    source = np.arange(6, dtype=np.float32)
    target = np.array([10., 20.])
    data = create_data(source, target=target,
                       data_type='time_series', ts_forecast_horizon=2)
    np.testing.assert_array_equal(data.features, source.reshape(1, -1))
    np.testing.assert_array_equal(data.target, target.reshape(1, -1))


def test_explicit_forecasting_task_infers_type_without_changing_dataspec_default():
    source = np.arange(8, dtype=np.float32)
    data = create_data(source, task='ts_forecasting', ts_forecast_horizon=2)
    assert data.data_type.value == 'time_series'
    np.testing.assert_array_equal(data.features, source[:6].reshape(1, -1))
    np.testing.assert_array_equal(data.target, source[6:].reshape(1, -1))
    assert DataSpec().data_type.value == 'table'


@pytest.mark.parametrize('position', [0, -1])
def test_labelled_2d_time_series_keep_explicit_target_column_convention(position):
    source = np.arange(18, dtype=np.float32).reshape(3, 6)
    train = create_data(source, data_type='time_series', target_idx=position)
    expected = np.delete(source, position, axis=1)
    np.testing.assert_array_equal(train.features, expected)
    np.testing.assert_array_equal(train.target[:, 0], source[:, position])
    predicted = create_data(expected, from_data=train)
    torch.testing.assert_close(predicted.features, train.features)
    assert predicted.idx_mapping == train.idx_mapping


def test_all_missing_targets_fail_before_any_fit(monkeypatch):
    from fedot.core.data.tensor_data.preparation import PreparationRuntime

    def forbidden(*args, **kwargs):
        raise AssertionError('empty rows reached fit')

    monkeypatch.setattr(PreparationRuntime, 'fit', forbidden)
    with pytest.raises(TensorDataContractError) as error:
        create_data(np.ones((3, 2)), target=np.full(3, np.nan))
    assert error.value.code == 'missing_target'


def test_backend_is_restored_on_success_and_error(monkeypatch):
    backend = Backend()
    backend.set('cpu')
    initial = (backend.name, backend.device, backend.xp, backend.pd)
    changes = []
    original = Backend._set_backend

    def record(self, name):
        changes.append(name)
        return original(self, name)

    monkeypatch.setattr(Backend, '_set_backend', record)
    create_data(np.ones((2, 2)), target=np.array([0, 1]), use_cache=False)
    assert (backend.name, backend.device, backend.xp, backend.pd) == initial
    assert len(changes) >= 2
    with pytest.raises(TensorDataContractError):
        create_data(np.ones((2, 2)), idx=[0])
    assert (backend.name, backend.device, backend.xp, backend.pd) == initial


def test_lazy_options_are_snapshotted_and_materialization_is_once():
    options = {'batch_size': 8}
    lazy = create_data_lazy(np.ones((2, 2)), target=np.array(
        [0, 1]), dataloader_kwargs=options)
    options['batch_size'] = 64
    assert lazy._data is None
    first = lazy.get()
    assert lazy.get() is first
    assert first.dataloader_kwargs['batch_size'] == 8


def test_legacy_trace_reuses_encoding_without_target_handler_corruption():
    train = create_data(
        np.array([[10., 'a', 'no'], [20., 'b', 'yes']], dtype=object))
    test = TensorDataCreator.create(np.array([[30., 'b']], dtype=object), 'cpu',
                                    state=StateEnum.PREDICT, trace_uuid=train.trace_uuid)
    np.testing.assert_array_equal(test.features, [[30., 1.]])
    assert test.target is None


def test_cached_fit_does_not_reuse_another_calls_row_labels():
    features = np.array([[1., 'a'], [2., 'b']], dtype=object)
    target = np.array([0, 1])
    first = create_data(features, target=target, idx=['first', 'second'])
    second = create_data(features, target=target, idx=['third', 'fourth'])
    np.testing.assert_array_equal(second.idx, ['third', 'fourth'])
    assert first.trace_uuid is not None
    assert second.trace_uuid is not None
    third = create_data(features, from_data=second)
    torch.testing.assert_close(first.features, third.features)


def test_identical_encoded_tensors_cannot_substitute_another_fitted_vocabulary():
    target = np.array([0, 1])
    for vocabulary in (['a', 'b'], ['x', 'y'], ['x', 'y'], ['a', 'b']):
        source = np.array(vocabulary, dtype=object).reshape(-1, 1)
        train = create_data(source, target=target)
        predicted = create_data(source, from_data=train)
        np.testing.assert_array_equal(predicted.features[:, 0], [0, 1])
