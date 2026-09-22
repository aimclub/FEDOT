from types import SimpleNamespace

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

    assert torch.equal(result.features, torch.tensor(
        [[10.0, 20.0], [30.0, 40.0]]))
    assert result.predict is None
    assert torch.equal(result.target, tensor_data.target)


@pytest.mark.unit
def test_is_tensor_transform_operation_for_model_and_transform():
    assert Model('linear')._is_tensor_transform_operation() is False
    assert Model('torch_linear')._is_tensor_transform_operation() is False
    assert DataOperation('pca')._is_tensor_transform_operation() is True


@pytest.mark.unit
@pytest.mark.parametrize('is_transform', [False, True])
def test_operation_predict_preserves_strategy_tensor_output(
        tensor_data, monkeypatch, is_transform):
    if is_transform:
        operation = DataOperation('pca')
        expected = EvaluationStrategy._replace_features_in_tensor_data(
            torch.tensor([[10.0, 20.0], [30.0, 40.0]]), tensor_data)
    else:
        operation = Model('torch_linear')
        expected = EvaluationStrategy._replace_predict_in_tensor_data(
            torch.tensor([0.2, 0.8]), tensor_data)
    calls = []
    fitted = object()

    def predict(trained_operation, predict_data):
        calls.append((trained_operation, predict_data))
        return expected

    operation._eval_strategy = SimpleNamespace(predict=predict)
    monkeypatch.setattr(operation, '_init', lambda *args, **kwargs: None)

    result = operation.predict(fitted, tensor_data)

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is fitted
    assert calls[0][1] is tensor_data
    assert torch.equal(tensor_data.features,
                       torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(tensor_data.predict, torch.tensor([0.5, 0.6]))
