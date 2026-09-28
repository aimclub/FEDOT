"""Focused row, mask, ownership and transfer contract regressions."""
import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.backend.backend import Backend
from fedot.core.data.bridges.input_to_tensor import input_data_to_tensordata
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data
from fedot.core.data.input_data.data import InputData
from fedot.core.data.tensor_data.contracts import TensorDataContractError, column_indices, select_rows
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import drop_rows_with_nan, target_row_mask
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


pytestmark = pytest.mark.unit


def test_boolean_sample_mask_reduces_all_non_sample_axes():
    values = torch.ones((3, 2, 4))
    values[1, 1, 2] = torch.nan
    original = values.clone()
    kept, count = drop_rows_with_nan(values)
    assert count == 1
    assert kept.shape == (2, 2, 4)
    torch.testing.assert_close(kept, original[[0, 2]])
    torch.testing.assert_close(values, original, equal_nan=True)
    np.testing.assert_array_equal(target_row_mask(
        values.numpy()), [True, False, True])


@pytest.mark.parametrize('mask', [[1, 0, 1], [[True, False, True]], [True]])
def test_sample_mask_rejects_wrong_dtype_rank_or_length(mask):
    with pytest.raises(TensorDataContractError) as error:
        select_rows(np.arange(3), mask)
    assert error.value.code == 'invalid_mask'


def test_column_positions_are_canonical_and_never_mutated():
    selectors = [-1, 0]
    assert column_indices(selectors, 3) == [2, 0]
    assert selectors == [-1, 0]
    with pytest.raises(TensorDataContractError):
        column_indices([0, -3], 3)


def test_direct_container_copies_metadata_but_explicitly_borrows_tensors():
    features = torch.ones((3, 2))
    labels = np.array([8, 7, 6])
    mapping = {0: 1, 1: 2}
    options = {'nested': {'value': 1}}
    data = TensorData('classification', 'tabular', features,
                      idx=labels, idx_mapping=mapping, dataloader_kwargs=options)
    assert data.features is features
    data.idx[0] = 100
    data.idx_mapping[0] = 9
    data.dataloader_kwargs['nested']['value'] = 2
    assert labels[0] == 8
    assert mapping[0] == 1
    assert options['nested']['value'] == 1


@pytest.mark.parametrize('field,value', [('idx', [0]), ('target', torch.ones(2)), ('predict', torch.ones(2))])
def test_direct_container_rejects_misaligned_row_fields(field, value):
    with pytest.raises(TensorDataContractError) as error:
        TensorData('classification', 'tabular',
                   torch.ones((3, 2)), **{field: value})
    assert error.value.code == 'row_alignment'


def test_to_moves_all_tensor_fields_without_changing_backend(monkeypatch):
    data = TensorData('classification', 'tabular', torch.ones((3, 2)),
                      target=torch.ones(3), predict=torch.ones(3), idx=torch.arange(3))
    moved = []
    old = Backend().name
    original = torch.Tensor.to

    def record(tensor, device, *args, **kwargs):
        moved.append(id(tensor))
        return original(tensor, device, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', record)
    assert data.to('cpu') is data
    assert len(moved) == 4
    assert Backend().name == old
    assert data.device == torch.device('cpu')


def test_validate_does_not_copy_tensor_storage(monkeypatch):
    data = TensorData('classification', 'tabular', torch.ones((3, 2)),
                      target=torch.ones(3), idx=torch.arange(3))

    def forbid_copy(*args, **kwargs):
        raise AssertionError(
            'validation must inspect shape without copying tensor storage')

    monkeypatch.setattr(torch.Tensor, 'clone', forbid_copy)
    assert data.validate() is data


def test_failed_transfer_is_atomic(monkeypatch):
    data = TensorData('classification', 'tabular',
                      torch.ones((3, 2)), target=torch.ones(3))
    features, target = data.features, data.target

    def transfer(tensor, *args, **kwargs):
        if tensor is target:
            raise RuntimeError('simulated allocation failure')
        return tensor.clone()

    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    with pytest.raises(RuntimeError, match='allocation'):
        data.to('cpu')
    assert data.features is features
    assert data.target is target


def test_input_bridge_preserves_sample_ids_through_target_mask():
    data = InputData(idx=np.array([9, 5, 7]), features=np.ones((3, 2)),
                     target=np.array([0., np.nan, 1.]), task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    tensor = input_data_to_tensordata(data, 'cpu')
    np.testing.assert_array_equal(tensor.idx, [9, 7])
    restored = tensordata_to_input_data(tensor)
    np.testing.assert_array_equal(restored.idx, [9, 7])
    restored.features[0, 0] = 99
    assert tensor.features[0, 0] == 1


def test_names_and_mapping_after_equal_width_one_hot_expansion():
    from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum

    source = np.array([['a', 10, 'x'], ['b', 20, 'x']], dtype=object)
    data = create_data(source, target=np.array([0, 1]), features_names=['a', 'b', 'c'],
                       encoding_strategy=[{'method': EncodingMethodEnum.ohe, 'features_idx': [0, 2]}])
    assert data.features.shape == (2, 4)
    assert data.idx_mapping == {0: 1, 1: 0, 2: 0, 3: 2}
    assert data.categorical_idx == [1, 2, 3]
    assert data.numerical_idx == [0]
    assert data.features_names == ['b', 'a__0', 'a__1', 'c']
    predict = create_data(source, from_data=data,
                          features_names=['a', 'b', 'c'])
    torch.testing.assert_close(data.features, predict.features)


def test_dataframe_one_hot_bridge_roundtrip_has_unique_prepared_names():
    from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum

    source = pd.DataFrame(
        {'a': ['red', 'blue', 'red'], 'a__0': [10., 20., 30.]})
    data = create_data(source, target=np.array([0, 1, 0]), idx=['x', 'z', 'y'],
                       encoding_strategy=[{'method': EncodingMethodEnum.ohe, 'features_idx': ['a']}])
    assert data.preparation_state.schema.names == ('a', 'a__0')
    assert data.features_names == ['a__0', 'a__1', 'a__2']
    assert len(set(data.features_names)) == data.features.shape[1]
    restored = input_data_to_tensordata(tensordata_to_input_data(data), 'cpu')
    assert restored.features_names == data.features_names
    torch.testing.assert_close(restored.features, data.features)
    torch.testing.assert_close(restored.target, data.target)
    np.testing.assert_array_equal(restored.idx, data.idx)


def test_equal_width_expansion_still_reorders_column_ownership():
    from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum

    source = np.array([['a', 10], ['a', 20]], dtype=object)
    data = create_data(source, target=np.array([0, 1]),
                       encoding_strategy=[{'method': EncodingMethodEnum.ohe, 'features_idx': [0]}])
    assert data.features.shape == (2, 2)
    assert data.idx_mapping == {0: 1, 1: 0}
    assert data.categorical_idx == [1]


def test_legacy_temporal_indices_are_explicitly_not_sample_indices():
    data = InputData(idx=np.arange(5), features=np.arange(5), target=None,
                     task=Task(TaskTypesEnum.ts_forecasting), data_type=DataTypesEnum.ts)
    with pytest.raises(TensorDataContractError) as error:
        input_data_to_tensordata(data, 'cpu')
    assert error.value.code == 'temporal_index_bridge'
