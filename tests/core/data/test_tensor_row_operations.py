"""Device-local row algorithms and native stratified split contracts."""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from fedot.core.data.merge.data_merger import TensorDataMerger
from fedot.core.data.split.data_split import train_test_tensor_data_setup
from fedot.core.data.split.tensor_split import stratified_split_indices
from fedot.core.data.tensor_data.contracts import TensorDataContractError, select_rows
from fedot.core.data.tensor_data.row_operations import common_row_indices, row_positions, take_rows
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import td_values_equal
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


pytestmark = pytest.mark.unit


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return request.param


@contextmanager
def no_host_arrays(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A tensor was moved through CPU/NumPy')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(torch.Tensor, 'numpy', forbidden)
        patch.setattr(torch.Tensor, 'tolist', forbidden)
        yield


@pytest.mark.parametrize('target', [
    [0] * 50 + [1] * 30 + [2] * 20,
    [0] * 7 + [1] * 11 + [2] * 13,
    [[0, 1]] * 10 + [[1, 0]] * 10 + [[1, 1]] * 10,
])
def test_stratification_is_proportional_disjoint_and_reproducible(target, device, monkeypatch):
    target = torch.tensor(target, device=device)
    train_size = int(len(target) * .8)
    global_state = torch.random.get_rng_state().clone()
    with no_host_arrays(monkeypatch):
        first = stratified_split_indices(target, train_size, random_seed=17)
        repeated = stratified_split_indices(target, train_size, random_seed=17)
        different = stratified_split_indices(target, train_size, random_seed=18)
    train, test = first
    assert train.device == test.device == target.device
    assert len(train) == train_size and len(test) == len(target) - train_size
    assert torch.equal(torch.cat(first).sort().values, torch.arange(len(target), device=device))
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))
    assert not torch.equal(train, different[0])
    assert torch.equal(torch.random.get_rng_state(), global_state)
    _, counts = torch.unique(target, dim=0, return_counts=True)
    _, actual = torch.unique(target[train], dim=0, return_counts=True)
    assert torch.all((actual - counts * train_size / len(target)).abs() < 1)


@pytest.mark.parametrize('target, train_size', [([0] * 9 + [1], 8), ([0, 0, 1, 1, 2, 2], 2)])
def test_unsupported_strata_keep_sequential_fallback(target, train_size, device, monkeypatch):
    target = torch.tensor(target, device=device)
    data = TensorData(Task(TaskTypesEnum.classification), DataTypesEnum.tabular,
                      torch.arange(len(target), device=device).reshape(-1, 1).float(),
                      target=target, idx=torch.arange(len(target), device=device))
    with no_host_arrays(monkeypatch):
        assert stratified_split_indices(target, train_size) is None
        train, test = train_test_tensor_data_setup(data, train_size / len(target), stratify=True)
    assert torch.equal(train.idx, torch.arange(train_size, device=device))
    assert torch.equal(test.idx, torch.arange(train_size, len(target), device=device))


def test_tensor_selection_alignment_and_comparison_do_not_use_host_arrays(device, monkeypatch):
    labels = torch.tensor([9, 2, 7, 2], device=device)
    selected = torch.tensor([2, 9], device=device)
    values = torch.tensor([[90.], [20.], [70.], [21.]], device=device)
    with no_host_arrays(monkeypatch):
        positions = row_positions(labels, selected)
        result = take_rows(values, positions)
        common = common_row_indices([labels, torch.tensor([7, 9], device=device)])
        masked = select_rows(values, torch.tensor([True, False, True, False], device=device))
        equal = td_values_equal(values, values.clone())
    assert torch.equal(positions, torch.tensor([3, 0], device=device))
    assert torch.equal(result, torch.tensor([[21.], [90.]], device=device))
    assert torch.equal(common, torch.tensor([7, 9], device=device))
    assert torch.equal(masked, values[[0, 2]])
    assert equal


def test_host_category_selection_and_mask_validation(device):
    categories = np.array([['red'], ['blue'], ['green']], dtype=object)
    selected = take_rows(categories, torch.tensor([2, 0], device=device))
    assert selected[:, 0].tolist() == ['green', 'red']
    assert take_rows(None, torch.tensor([], dtype=torch.long, device=device)) is None
    for mask in (torch.ones(3, dtype=torch.long, device=device), torch.ones(3, 1, dtype=torch.bool, device=device)):
        with pytest.raises(TensorDataContractError, match='one boolean per sample'):
            select_rows(torch.ones(3, 2, device=device), mask)


def test_tensor_subset_rows_repeats_all_row_fields_and_invalidates_fingerprint(device):
    values = torch.arange(6., device=device).reshape(3, 2)
    source = TensorData(Task(TaskTypesEnum.classification), DataTypesEnum.tabular, values,
                        target=torch.tensor([0, 1, 0], device=device), idx=['a', 'b', 'c'],
                        predict=values[:, :1], categorical_features=np.array([['red'], ['blue'], ['green']]),
                        categorical_idx=[1], numerical_idx=[0], categorical_source_idx=[7],
                        idx_mapping={0: 5, 1: 7}, features_names=['value', 'color'], fingerprint='original')
    selected = source.subset_rows(torch.tensor([2, 0, 2], device=device))
    torch.testing.assert_close(selected.features, values[[2, 0, 2]])
    torch.testing.assert_close(selected.target, source.target[[2, 0, 2]])
    torch.testing.assert_close(selected.predict, source.predict[[2, 0, 2]])
    assert selected.idx == ['c', 'a', 'c']
    assert selected.categorical_features[:, 0].tolist() == ['green', 'red', 'green']
    assert selected.idx_mapping == source.idx_mapping
    assert selected.features_names == source.features_names
    assert selected.fingerprint is None
    assert source.fingerprint == 'original'
    selected.features.zero_()
    torch.testing.assert_close(source.features, torch.arange(6., device=device).reshape(3, 2))


@pytest.mark.parametrize('host_labels', [False, True])
def test_split_and_merge_align_tensor_rows_without_host_roundtrip(device, monkeypatch, host_labels):
    left = TensorData(Task(TaskTypesEnum.classification), DataTypesEnum.tabular,
                      torch.tensor([[90.], [20.], [70.]], device=device),
                      idx=torch.tensor([9, 2, 7], device=device), target=torch.tensor([1, 0, 1], device=device))
    right = TensorData(Task(TaskTypesEnum.classification), DataTypesEnum.tabular,
                       torch.tensor([[700.], [900.], [300.]], device=device),
                       idx=np.array([7, 9, 3]) if host_labels else torch.tensor([7, 9, 3], device=device))
    with no_host_arrays(monkeypatch):
        merged = TensorDataMerger([left, right], [False, False]).merge()
    assert torch.equal(merged.idx, torch.tensor([7, 9], device=device))
    assert torch.equal(merged.features, torch.tensor([[70., 700.], [90., 900.]], device=device))
    assert torch.equal(merged.target, torch.tensor([1, 1], device=device))


def test_stratified_tensordata_split_keeps_all_row_fields_on_device(device, monkeypatch):
    target = torch.arange(90, device=device) % 3
    data = TensorData(Task(TaskTypesEnum.classification), DataTypesEnum.tabular,
                      torch.arange(90., device=device).reshape(-1, 1), target=target,
                      idx=torch.arange(90, device=device), predict=torch.arange(90., device=device))
    with no_host_arrays(monkeypatch):
        train, test = train_test_tensor_data_setup(data, stratify=True)
    for part in (train, test):
        assert part.device == data.device
        assert torch.equal(part.idx.float(), part.features[:, 0])
        assert torch.equal(part.predict, part.idx.float())
        assert torch.equal(part.target, target[part.idx])
