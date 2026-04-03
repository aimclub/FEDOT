import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.industrial_nn_bridge import IndustrialNNBridgeStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_input_data():
    return InputData(
        idx=np.arange(4),
        features=np.arange(8).reshape(4, 2),
        target=np.array([0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_industrial_nn_bridge_strategy_uses_resolved_model_and_normalizes_labels(monkeypatch):
    captured = {}

    class FakeIndustrialModel:
        def __init__(self, params):
            captured['params'] = params.to_dict()

        def fit(self, input_data):
            captured['fit_input'] = input_data
            return self

        def predict(self, input_data, output_mode='default'):
            captured['predict_input'] = input_data
            return OutputData(
                idx=input_data.idx,
                predict=np.array([[0.2, 0.8], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]]),
                target=input_data.target,
                task=input_data.task,
                data_type=DataTypesEnum.table,
            )

    monkeypatch.setattr(
        'fedot.core.operations.evaluation.industrial_nn_bridge.resolve_industrial_nn_model_class',
        lambda operation_type: FakeIndustrialModel,
    )

    strategy = IndustrialNNBridgeStrategy('industrial_inception_nn', OperationParameters(epochs=3))
    strategy.output_mode = 'labels'
    input_data = _make_input_data()

    fitted = strategy.fit(input_data)
    prediction = strategy.predict(fitted, input_data)

    assert captured['params']['epochs'] == 3
    assert captured['params']['batch_size'] == 32
    assert np.array_equal(prediction.predict, np.array([1, 0, 1, 0]))

from pathlib import Path


def test_industrial_bridge_transitive_torch_modules_defer_annotations_for_py39():
    files = [
        Path('fedot/industrial/core/operation/transformation/torch_backend/statistical/stat_features.py'),
        Path('fedot/industrial/core/operation/transformation/basis/eigen_basis_torch.py'),
    ]

    for path in files:
        source = path.read_text(encoding='utf-8').splitlines()
        assert 'from __future__ import annotations' in source[:3]