import numpy as np

from fedot.core.data.data import InputData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_input_data():
    return InputData(
        idx=np.arange(6),
        features=np.arange(12).reshape(6, 2),
        target=np.array([0, 1, 0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_data_source_splitter_build_tensordata_uses_bridge_boundary(monkeypatch):
    splitter = DataSourceSplitter(cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    converted_input = _make_input_data()
    captured = {}

    def fake_to_input_data(tensor_data):
        captured['tensor_data'] = tensor_data
        return converted_input

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.tensordata_to_input_data',
        fake_to_input_data,
    )

    producer = splitter.build_tensordata('tensor-data')
    train_data, test_data = next(producer())

    assert captured['tensor_data'] == 'tensor-data'
    assert isinstance(train_data, InputData)
    assert isinstance(test_data, InputData)
    assert train_data.task.task_type is TaskTypesEnum.classification
    assert test_data.task.task_type is TaskTypesEnum.classification


def test_data_source_splitter_build_tensordata_keeps_input_data_shape(monkeypatch):
    splitter = DataSourceSplitter(cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    converted_input = _make_input_data()

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.tensordata_to_input_data',
        lambda tensor_data: converted_input,
    )

    producer = splitter.build_tensordata('tensor-data')
    train_data, test_data = next(producer())

    assert train_data.features.shape[1] == converted_input.features.shape[1]
    assert test_data.features.shape[1] == converted_input.features.shape[1]
    assert train_data.target is not None
    assert test_data.target is not None
