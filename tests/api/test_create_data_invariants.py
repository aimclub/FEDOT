"""Deterministic conservation and batching laws at the create_data boundary."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.data.tensor_data.contracts import TensorDataContractError
from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum


pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('isolated_cache_dir')]


def raw_source(values, kind, names=None):
    if kind == 'torch':
        return torch.from_numpy(values.copy())
    if kind == 'dataframe':
        return pd.DataFrame(values.copy(), columns=names)
    return values.copy()


def source_values(source):
    if isinstance(source, pd.DataFrame):
        return source.to_numpy()
    return np.asarray(source)


@pytest.mark.parametrize('kind', ['numpy', 'torch', 'dataframe'])
@pytest.mark.parametrize('width,outputs', [(1, 1), (3, 2), (5, 3)])
@pytest.mark.parametrize('missing_rows', [(), (0, 5), (1, 2, 4)])
def test_multioutput_mask_conserves_values_labels_and_owned_storage(kind, width, outputs, missing_rows):
    """Any missing output removes exactly that sample, not a feature or output."""
    values = (100 * np.arange(6)[:, None] +
              np.arange(width)).astype(np.float32)
    target = (10 * np.arange(6)[:, None] +
              np.arange(outputs)).astype(np.float32)
    for row in missing_rows:
        target[row, row % outputs] = np.nan
    expected_rows = [row for row in range(6) if row not in missing_rows]
    labels = np.array([90, -2, 90, 7, 0, 42])
    source = raw_source(values, kind)
    original_target = target.copy()
    result = create_data(source, target=target, idx=labels,
                         task='regression', use_cache=False)

    np.testing.assert_array_equal(result.features, values[expected_rows])
    np.testing.assert_array_equal(
        result.target, original_target[expected_rows])
    np.testing.assert_array_equal(result.idx, labels[expected_rows])
    assert result.features.shape == (len(expected_rows), width)
    assert result.target.shape == (len(expected_rows), outputs)
    assert result.idx_mapping == dict(enumerate(range(width)))

    result.features[0, 0] = -123
    result.target[0, 0] = -456
    result.idx[0] = -789
    np.testing.assert_array_equal(source_values(source), values)
    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(labels, [90, -2, 90, 7, 0, 42])


@pytest.mark.parametrize('kind', ['numpy', 'torch', 'dataframe'])
@pytest.mark.parametrize('positions', [(4, 1), (0, 3), (2,)])
@pytest.mark.parametrize('selector_kind', ['positive', 'negative', 'names'])
def test_target_split_can_reconstruct_source_columns_in_original_order(kind, positions, selector_kind):
    """Target order follows selectors; feature mapping follows surviving source columns."""
    values = np.arange(30, dtype=np.float32).reshape(6, 5)
    names = ['left', 'label_a', 'middle', 'right', 'label_b']
    source = raw_source(values, kind, names)
    selectors = list(positions)
    if selector_kind == 'negative':
        selectors = [position - 5 for position in positions]
    elif selector_kind == 'names':
        selectors = [names[position] for position in positions]
    original_selectors = selectors.copy()
    kept = [position for position in range(5) if position not in positions]
    labels = np.array(['z', 'a', 'z', 'q', 'r', 's'])
    result = create_data(source, target_idx=selectors, features_names=names,
                         idx=labels, task='regression', use_cache=False)

    rebuilt = np.empty_like(values)
    for current, original in result.idx_mapping.items():
        rebuilt[:, original] = result.features[:, current]
    for current, original in enumerate(positions):
        rebuilt[:, original] = result.target[:, current]
    np.testing.assert_array_equal(rebuilt, values)
    np.testing.assert_array_equal(result.idx, labels)
    assert result.idx_mapping == dict(enumerate(kept))
    assert result.features_names == [names[position] for position in kept]
    assert selectors == original_selectors
    assert names == ['left', 'label_a', 'middle', 'right', 'label_b']

    predicted = create_data(raw_source(values[:, kept], kind, result.features_names),
                            features_names=result.features_names, from_data=result, use_cache=False)
    np.testing.assert_array_equal(predicted.features, values[:, kept])
    assert predicted.idx_mapping == dict(enumerate(kept))


@pytest.mark.parametrize('selectors,code', [
    ([4, -1], 'duplicate_selector'),
    ([-5, 0], 'duplicate_selector'),
    (['b', 'b'], 'duplicate_selector'),
    (np.array([False, True, False, True, False]), 'invalid_selector'),
    (torch.tensor([True, False, False, False, True]), 'invalid_selector'),
])
def test_selector_alias_failures_preserve_inputs_and_identify_public_field(selectors, code):
    source = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'))
    before = source.copy(deep=True)
    with pytest.raises(TensorDataContractError) as error:
        create_data(source, target_idx=selectors, use_cache=False)
    assert (error.value.code, error.value.field) == (code, 'target_idx')
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize('method', [EncodingMethodEnum.label, EncodingMethodEnum.ohe])
@pytest.mark.parametrize('partitions', [(1, 2, 3), (3, 1, 2), (2, 2, 2)])
def test_fitted_categorical_codes_are_independent_of_batches_and_interleaving(method, partitions):
    """Unknown-only batches cannot learn vocabulary or alter later known codes."""
    train_frame = pd.DataFrame({
        'color': ['c', 'a', 'b', 'discard'],
        'amount': [10., 20., 30., 40.],
        'group': ['y', 'x', 'x', 'discard'],
    })
    train = create_data(train_frame, target=np.array([1., 0., 1., np.nan]), use_cache=False,
                        encoding_strategy=[{'method': method, 'features_idx': ['color', 'group']}])
    fitted = deepcopy(train.preparation_state)
    train_features, train_target = train.features.clone(), train.target.clone()
    incoming = pd.DataFrame({
        'color': ['new', 'discard', 'a', 'c', 'b', 'new'],
        'amount': [101., 102., 103., 104., 105., 106.],
        'group': ['unknown', 'discard', 'x', 'y', 'x', 'y'],
    })
    before = incoming.copy(deep=True)
    color_codes = {'a': 0, 'b': 1, 'c': 2}
    group_codes = {'x': 0, 'y': 1}
    if method == EncodingMethodEnum.label:
        expected = np.array([
            [color_codes.get(color, 3), amount, group_codes.get(group, 2)]
            for color, amount, group in incoming.itertuples(index=False, name=None)
        ])
    else:
        expected = np.array([
            [amount, int(color == 'a'), int(color == 'b'), int(color == 'c'),
             int(group == 'x'), int(group == 'y')]
            for color, amount, group in incoming.itertuples(index=False, name=None)
        ])
    labels = np.array(['u', 'v', 'a', 'c', 'b', 'w'])
    whole = create_data(incoming, from_data=train, idx=labels, use_cache=False)
    np.testing.assert_array_equal(whole.features, expected)

    chunks = []
    start = 0
    for size in partitions:
        stop = start + size
        chunk = create_data(incoming.iloc[start:stop], from_data=train,
                            idx=labels[start:stop], use_cache=False)
        np.testing.assert_array_equal(chunk.features, expected[start:stop])
        assert chunk.target is None
        assert chunk.idx_mapping == train.idx_mapping
        chunks.append(chunk)
        known = create_data(
            train_frame.iloc[:3], from_data=train, use_cache=False)
        torch.testing.assert_close(known.features, train_features)
        assert train.preparation_state == fitted
        start = stop
    torch.testing.assert_close(
        torch.cat([chunk.features for chunk in chunks]), whole.features)
    np.testing.assert_array_equal(np.concatenate(
        [chunk.idx for chunk in chunks]), labels)
    torch.testing.assert_close(train.features, train_features)
    torch.testing.assert_close(train.target, train_target)
    pd.testing.assert_frame_equal(incoming, before)


@pytest.mark.parametrize('channels', [None, 2])
@pytest.mark.parametrize('horizon', [1, 3])
def test_forecasting_conserves_time_and_accepts_variable_prediction_context(channels, horizon):
    """A horizon is a final-axis suffix; prediction never splits its context."""
    shape = (4, 9) if channels is None else (4, channels, 9)
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    labels = np.array([80, -1, 80, 5])
    train = create_data(source, task='ts_forecasting', ts_forecast_horizon=horizon,
                        idx=labels, use_cache=False)
    np.testing.assert_array_equal(np.concatenate(
        [train.features, train.target], axis=-1), source)
    assert train.features.shape == shape[:-1] + (9 - horizon,)
    assert train.target.shape == shape[:-1] + (horizon,)
    assert train.ts_init_shape == train.features.shape
    fitted = deepcopy(train.preparation_state)

    for context in (1, 4, 11):
        prediction_shape = shape[:-1] + (context,)
        values = (np.arange(np.prod(prediction_shape)) +
                  200).astype(np.float32).reshape(prediction_shape)
        predicted = create_data(values, from_data=train,
                                idx=labels, use_cache=False)
        np.testing.assert_array_equal(predicted.features, values)
        np.testing.assert_array_equal(predicted.idx, labels)
        assert predicted.target is None
        assert predicted.ts_init_shape == prediction_shape
        assert predicted.idx_mapping == dict(
            enumerate(range(prediction_shape[1])))
        assert train.preparation_state == fitted


@pytest.mark.parametrize('kind', ['numpy', 'torch'])
@pytest.mark.parametrize('horizon', [1, 3])
def test_three_dimensional_forecast_target_mask_keeps_whole_samples(kind, horizon):
    """A missing horizon value in any channel removes its entire sample."""
    source = np.arange(6 * 2 * 8, dtype=np.float32).reshape(6, 2, 8)
    missing_rows = [1, 4]
    source[1, 0, -1] = np.nan
    source[4, 1, -horizon] = np.nan
    expected_rows = [0, 2, 3, 5]
    raw = raw_source(source, kind)
    labels = np.array(['f', 'e', 'd', 'c', 'b', 'a'])
    data = create_data(raw, task='ts_forecasting', ts_forecast_horizon=horizon,
                       idx=labels, use_cache=False)
    np.testing.assert_array_equal(
        data.features, source[expected_rows, :, :-horizon])
    np.testing.assert_array_equal(
        data.target, source[expected_rows, :, -horizon:])
    np.testing.assert_array_equal(data.idx, labels[expected_rows])
    assert data.features.shape[0] + len(missing_rows) == source.shape[0]
    assert data.idx_mapping == {0: 0, 1: 1}
    np.testing.assert_array_equal(source_values(raw), source)
