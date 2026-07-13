import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.model import Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture
def tensor_data():
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        target=torch.tensor([[0.0], [1.0]]),
        predict=torch.tensor([0.5, 0.6]),
    )


@pytest.mark.unit
def test_replace_predict_in_tensor_data_keeps_features(tensor_data):
    result = EvaluationStrategy._replace_predict_in_tensor_data(
        torch.tensor([0.2, 0.8]),
        tensor_data,
    )

    assert torch.equal(result.features, tensor_data.features)
    assert torch.equal(result.predict, torch.tensor([0.2, 0.8]))


@pytest.mark.unit
def test_replace_features_in_tensor_data_clears_predict(tensor_data):
    result = EvaluationStrategy._replace_features_in_tensor_data(
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        tensor_data,
    )

    assert torch.equal(result.features, torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
    assert result.predict is None
    assert torch.equal(result.target, tensor_data.target)


@pytest.mark.unit
def test_is_tensor_transform_operation_for_model_and_transform():
    assert Model('linear')._is_tensor_transform_operation() is False
    assert Model('torch_linear')._is_tensor_transform_operation() is False
    assert DataOperation('pca')._is_tensor_transform_operation() is True


@pytest.mark.unit
def test_operation_wraps_tensor_result_by_operation_kind(tensor_data):
    model = Model('torch_linear')
    transform = DataOperation('pca')

    model_output = model._wrap_tensor_operation_result(torch.tensor([0.2, 0.8]), tensor_data)
    transform_output = transform._wrap_tensor_operation_result(
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        tensor_data,
    )

    assert torch.equal(model_output.predict, torch.tensor([0.2, 0.8]))
    assert torch.equal(model_output.features, tensor_data.features)
    assert transform_output.predict is None
    assert torch.equal(transform_output.features, torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
