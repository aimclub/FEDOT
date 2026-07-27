import numpy as np
import torch

from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_tensor_data():
    return TensorData(
        state=StateEnum.FIT,
        features=torch.arange(12, dtype=torch.float32).reshape(6, 2),
        target=torch.tensor([0, 1, 0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=np.arange(6),
    )


def test_data_source_splitter_build_uses_tensor_holdout_setup(monkeypatch):
    splitter = DataSourceSplitter(
        cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    tensor_data = _make_tensor_data()
    train_data = _make_tensor_data()
    test_data = _make_tensor_data()
    captured = {}

    def fake_train_test_tensor_data_setup(data, split_ratio):
        captured['tensor_data'] = data
        captured['split_ratio'] = split_ratio
        return train_data, test_data

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.train_test_tensor_data_setup',
        fake_train_test_tensor_data_setup,
    )

    producer = splitter.build(tensor_data)
    produced_train, produced_test = next(producer())

    assert captured['tensor_data'] is tensor_data
    assert captured['split_ratio'] == 0.5
    assert produced_train is train_data
    assert produced_test is test_data


def test_data_source_splitter_build_keeps_tensor_data_shape(monkeypatch):
    splitter = DataSourceSplitter(
        cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    tensor_data = _make_tensor_data()
    train_data = _make_tensor_data()
    test_data = _make_tensor_data()

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.train_test_tensor_data_setup',
        lambda data, split_ratio: (train_data, test_data),
    )

    producer = splitter.build(tensor_data)
    produced_train, produced_test = next(producer())

    assert produced_train.features.shape[1] == tensor_data.features.shape[1]
    assert produced_test.features.shape[1] == tensor_data.features.shape[1]
    assert produced_train.target is not None
    assert produced_test.target is not None
