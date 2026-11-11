"""Data operations preserve the distinction between training and inference."""
from dataclasses import replace

import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.data_operations.common_preprocessing import PreprocessingDataOperationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class TrainingRowFilter(TensorDataOperationImplementation):
    def fit(self, data):
        return self

    def transform(self, data):
        return replace(data, features=data.features + 1, predict=None)

    def transform_for_fit(self, data):
        return replace(data, features=data.features[:1], target=data.target[:1], idx=data.idx[:1], predict=None)


@pytest.mark.unit
def test_data_operation_calls_implementation_training_transform(monkeypatch):
    data = TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        target=torch.tensor([[0.0], [1.0]]),
        idx=[10, 20],
    )
    # Substitute a fit-stage-specific implementation to exercise the shared dispatch contract.
    monkeypatch.setitem(PreprocessingDataOperationStrategy._operations_by_types, 'pca', TrainingRowFilter)
    operation = DataOperation('pca')

    fitted, training_output = operation.fit(OperationParameters(), data)
    inference_output = operation.predict(fitted, data)

    torch.testing.assert_close(training_output.features, data.features[:1])
    torch.testing.assert_close(training_output.target, data.target[:1])
    assert list(training_output.idx) == [10]
    torch.testing.assert_close(inference_output.features, data.features + 1)
    assert list(inference_output.idx) == [10, 20]
    assert list(data.idx) == [10, 20]
    assert data.features.shape == (2, 2)
    assert data.target.shape == (2, 1)
