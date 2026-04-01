import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.tools import StateEnum
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.runtime_data_rules import RuntimeFoldData
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


def test_data_source_splitter_build_runtime_wraps_tensor_folds(monkeypatch):
    splitter = DataSourceSplitter(cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    converted_input = _make_input_data()
    captured = []

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.input_data_to_tensordata',
        lambda data, backend_name, state=StateEnum.FIT: captured.append((backend_name, state)) or f'tensor:{state.value}',
    )

    producer = splitter.build(converted_input, runtime_mode='tensor_gpu_bridge', tensor_backend_name='gpu')
    train_data, test_data = next(producer())

    assert isinstance(train_data, RuntimeFoldData)
    assert isinstance(test_data, RuntimeFoldData)
    assert train_data.runtime_mode == 'tensor_gpu_bridge'
    assert test_data.runtime_mode == 'tensor_gpu_bridge'
    assert train_data.tensor_data == 'tensor:fit'
    assert test_data.tensor_data == 'tensor:predict'
    assert captured == [('gpu', StateEnum.FIT), ('gpu', StateEnum.PREDICT)]


def test_data_source_splitter_build_runtime_wraps_input_gpu_bridge_without_tensor(monkeypatch):
    splitter = DataSourceSplitter(cv_folds=None, split_ratio=0.5, shuffle=False, stratify=False)
    converted_input = _make_input_data()
    called = {'tensor': False}

    monkeypatch.setattr(
        'fedot.core.optimisers.objective.data_source_splitter.input_data_to_tensordata',
        lambda *args, **kwargs: called.update(tensor=True),
    )

    producer = splitter.build(converted_input, runtime_mode='input_gpu_bridge', tensor_backend_name='gpu')
    train_data, test_data = next(producer())

    assert isinstance(train_data, RuntimeFoldData)
    assert isinstance(test_data, RuntimeFoldData)
    assert train_data.tensor_data is None
    assert test_data.tensor_data is None
    assert called['tensor'] is False

