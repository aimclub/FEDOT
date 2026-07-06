from dataclasses import replace

import numpy as np
import pytest
import torch

from fedot.core.data.merge.data_merger import TensorDataMerger
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture
def classification_task():
    return Task(TaskTypesEnum.classification)


@pytest.fixture
def base_tensor_data(classification_task):
    return TensorData(
        task=classification_task,
        data_type=DataTypesEnum.tabular,
        idx=np.array([0, 1, 2, 3]),
        features=torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
        target=torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    )


@pytest.mark.unit
def test_tensor_data_merger_concatenates_parent_features(base_tensor_data):
    branch_a = replace(base_tensor_data, features=base_tensor_data.features[:, :1] * 10)
    branch_b = replace(base_tensor_data, features=base_tensor_data.features[:, 1:2] + 100)

    merged = TensorDataMerger([branch_a, branch_b]).merge()

    assert merged.features.shape == (4, 2)
    assert torch.allclose(merged.features[:, 0], branch_a.features[:, 0])
    assert torch.allclose(merged.features[:, 1], branch_b.features[:, 0])


@pytest.mark.unit
def test_tensor_data_merger_clears_predict_and_keeps_target(base_tensor_data):
    branch_a = replace(
        base_tensor_data,
        features=base_tensor_data.features[:, :1],
        predict=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    branch_b = replace(
        base_tensor_data,
        features=base_tensor_data.features[:, 1:2],
        predict=torch.tensor([0.5, 0.6, 0.7, 0.8]),
    )

    merged = TensorDataMerger([branch_a, branch_b]).merge()

    assert merged.predict is None
    assert torch.equal(merged.target, base_tensor_data.target)


@pytest.mark.unit
def test_tensor_data_merger_uses_branch_with_target_as_main(base_tensor_data):
    branch_without_target = replace(
        base_tensor_data,
        target=None,
        features=base_tensor_data.features[:, :1],
    )
    branch_with_target = replace(
        base_tensor_data,
        features=base_tensor_data.features[:, 1:2],
    )

    merged = TensorDataMerger([branch_without_target, branch_with_target]).merge()

    assert torch.equal(merged.target, branch_with_target.target)
    assert merged.features.shape == (4, 2)


@pytest.mark.unit
def test_tensor_data_merger_filters_by_common_idx(base_tensor_data):
    branch_a = replace(
        base_tensor_data,
        idx=np.array([0, 1, 2, 3]),
        features=torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
    )
    branch_b = replace(
        base_tensor_data,
        idx=np.array([1, 2, 3, 4]),
        features=torch.tensor([[101.0], [102.0], [103.0], [104.0]]),
    )

    merged = TensorDataMerger([branch_a, branch_b]).merge()

    assert np.array_equal(merged.idx, np.array([1, 2, 3]))
    assert merged.features.shape == (3, 2)
    assert torch.allclose(merged.features[:, 0], torch.tensor([20.0, 30.0, 40.0]))
    assert torch.allclose(merged.features[:, 1], torch.tensor([101.0, 102.0, 103.0]))
    assert torch.allclose(merged.target, torch.tensor([[1.0], [0.0], [1.0]]))


@pytest.mark.unit
def test_tensor_data_merger_single_parent_is_pass_through(base_tensor_data):
    branch = replace(
        base_tensor_data,
        features=base_tensor_data.features * 2,
        predict=torch.tensor([0.9, 0.8, 0.7, 0.6]),
    )

    merged = TensorDataMerger([branch]).merge()

    assert torch.equal(merged.features, branch.features)
    assert merged.predict is None


@pytest.mark.unit
def test_tensor_data_merger_raises_on_different_row_counts(base_tensor_data):
    branch_a = replace(base_tensor_data, idx=None, features=base_tensor_data.features[:2])
    branch_b = replace(base_tensor_data, idx=None, features=base_tensor_data.features[:3])

    with pytest.raises(ValueError, match='different row counts'):
        TensorDataMerger([branch_a, branch_b]).merge()


@pytest.mark.unit
def test_tensor_data_merger_raises_on_no_common_indices(base_tensor_data):
    branch_a = replace(base_tensor_data, idx=np.array([0, 1]))
    branch_b = replace(base_tensor_data, idx=np.array([2, 3]))

    with pytest.raises(ValueError, match='no common indices'):
        TensorDataMerger([branch_a, branch_b]).merge()


@pytest.mark.unit
def test_tensor_data_merger_raises_on_mixed_data_types(base_tensor_data):
    branch_a = replace(base_tensor_data, data_type=DataTypesEnum.tabular)
    branch_b = replace(base_tensor_data, data_type=DataTypesEnum.ts)

    with pytest.raises(ValueError, match="Can't merge different TensorData data types"):
        TensorDataMerger([branch_a, branch_b]).merge()


@pytest.mark.unit
def test_tensor_data_merger_promotes_1d_features_to_2d(base_tensor_data):
    branch_a = replace(
        base_tensor_data,
        features=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )
    branch_b = replace(
        base_tensor_data,
        features=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    merged = TensorDataMerger([branch_a, branch_b]).merge()

    assert merged.features.shape == (4, 2)
    assert torch.allclose(merged.features[:, 0], torch.tensor([10.0, 20.0, 30.0, 40.0]))
    assert torch.allclose(merged.features[:, 1], torch.tensor([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.unit
def test_tensor_data_merger_flattens_branches_with_different_ndims(base_tensor_data):
    branch_2d = replace(
        base_tensor_data,
        features=torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    )
    branch_3d = replace(
        base_tensor_data,
        features=torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
    )

    merged = TensorDataMerger([branch_2d, branch_3d]).merge()

    assert merged.features.shape == (4, 8)
    assert torch.allclose(merged.features[:, :2], branch_2d.features)
    assert torch.allclose(merged.features[:, 2:], branch_3d.features.reshape(4, -1))


@pytest.mark.unit
def test_tensor_data_merger_flattens_incompatible_same_ndim_shapes(base_tensor_data):
    branch_a = replace(
        base_tensor_data,
        features=torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
    )
    branch_b = replace(
        base_tensor_data,
        features=torch.arange(12, dtype=torch.float32).reshape(4, 1, 3),
    )

    merged = TensorDataMerger([branch_a, branch_b]).merge()

    assert merged.features.shape == (4, 9)
    assert torch.allclose(merged.features[:, :6], branch_a.features.reshape(4, -1))
    assert torch.allclose(merged.features[:, 6:], branch_b.features.reshape(4, -1))


@pytest.mark.unit
def test_tensordata_from_parents_merges_stub_parent_outputs(base_tensor_data):
    parent_a = PipelineNode('scaling')
    parent_b = PipelineNode('normalization')

    parent_a.fit_tensordata = lambda tensor_data, predictions_cache=None, fold_id=None: replace(
        tensor_data,
        features=tensor_data.features[:, :1] * 10,
        predict=torch.tensor([999.0]),
    )
    parent_b.fit_tensordata = lambda tensor_data, predictions_cache=None, fold_id=None: replace(
        tensor_data,
        features=tensor_data.features[:, 1:2] + 100,
        predict=torch.tensor([888.0]),
    )

    child = PipelineNode('torch_linear', nodes_from=[parent_b, parent_a])
    merged = child._tensordata_from_parents(base_tensor_data, parent_operation='fit')

    ordered_parents = [node.operation.operation_type for node in child._nodes_from_with_fixed_order()]
    assert ordered_parents == ['normalization', 'scaling']
    assert merged.features.shape == (4, 2)
    assert merged.predict is None
    assert torch.allclose(merged.features[:, 0], base_tensor_data.features[:, 1] + 100)
    assert torch.allclose(merged.features[:, 1], base_tensor_data.features[:, 0] * 10)
