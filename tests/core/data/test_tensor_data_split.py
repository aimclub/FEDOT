"""Tensor hold-out preserves split proportions and aligns every row field."""
import numpy as np
import pytest
import torch

from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import train_test_data_setup, train_test_tensor_data_setup
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.parametrize('target, task', [
    (np.arange(30) % 3, TaskTypesEnum.classification),
    (np.array([0] * 29 + [1]), TaskTypesEnum.classification),
    (np.arange(30, dtype=float), TaskTypesEnum.regression),
])
def test_tensor_holdout_preserves_class_counts_and_category_rows(target, task):
    features = np.arange(60, dtype=np.float32).reshape(30, 2)
    labels = np.arange(30)[::-1] * 3
    classic = InputData.from_numpy(features, target, idx=labels, task=task.value, categorical_idx=[])
    tensor = TensorData(Task(task), 'tabular', torch.from_numpy(features), target=torch.from_numpy(target),
                        idx=labels, predict=torch.arange(30), numerical_idx=[0, 1],
                        categorical_features=np.arange(30, dtype=object).reshape(-1, 1), categorical_source_idx=[2])
    expected = train_test_data_setup(classic)
    actual = train_test_tensor_data_setup(tensor, stratify=True)
    for old, new in zip(expected, actual):
        assert len(new.idx) == len(old.idx)
        np.testing.assert_array_equal(np.unique(new.target, return_counts=True),
                                      np.unique(old.target, return_counts=True))
        positions = new.features[:, 0].to(torch.long).numpy() // 2
        np.testing.assert_array_equal(new.idx, labels[positions])
        np.testing.assert_array_equal(new.target, target[positions])
        assert new.categorical_features[:, 0].tolist() == positions.tolist()
        assert new.predict.tolist() == positions.tolist()
        new.validate()


def test_default_tensor_holdout_remains_sequential_for_classification():
    data = TensorData(Task(TaskTypesEnum.classification), 'tabular', torch.arange(20.).reshape(10, 2),
                      target=torch.arange(10) % 2, idx=torch.arange(10))
    train, test = train_test_tensor_data_setup(data)
    assert train.idx.tolist() == list(range(8))
    assert test.idx.tolist() == [8, 9]


def test_tensor_split_handles_prediction_only_rows():
    data = TensorData(Task(TaskTypesEnum.regression), 'tabular', torch.arange(20.).reshape(10, 2))
    train, test = train_test_tensor_data_setup(data)
    assert train.features.shape == (8, 2)
    assert test.features.shape == (2, 2)
    assert train.target is test.target is None
