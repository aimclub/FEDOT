from types import SimpleNamespace

import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.api.main import FedotIndustrial


class _DummyEncoder:
    def inverse_transform(self, values):
        return np.array(values) + 10

    def transform(self, values):
        return np.array(values) + 1


def test_industrial_main_abstract_predict_uses_rule_based_pipeline_path(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    predict_data = SimpleNamespace(
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        target=np.array([1, 2, 3, 4]),
    )
    industrial.predict_data = predict_data
    industrial.target_encoder = _DummyEncoder()
    industrial.manager = SimpleNamespace(
        solver=SimpleNamespace(predict=lambda data, mode: OutputData(
            idx=np.arange(4),
            predict=np.array([1, 2, 3, 4]),
            target=None,
            task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
            data_type=DataTypesEnum.ts,
        )),
        condition_check=SimpleNamespace(
            solver_have_target_encoder=lambda encoder: True,
            solver_is_fedot_class=lambda solver: False,
            solver_is_pipeline_class=lambda solver: True,
        ),
    )

    result = industrial._FedotIndustrial__abstract_predict(predict_data, 'labels')

    assert np.array_equal(result, np.array([13, 14]))
    assert np.array_equal(industrial.predict_data.target, np.array([11, 12, 13, 14]))


def test_industrial_main_metric_evaluation_loop_uses_rule_based_shape_and_encoder(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    industrial.target_encoder = _DummyEncoder()
    industrial.manager = SimpleNamespace(
        condition_check=SimpleNamespace(solver_have_target_encoder=lambda encoder: True),
    )
    captured = {}

    monkeypatch.setattr(
        'fedot.industrial.api.main.FEDOT_GET_METRICS',
        {'classification': lambda **kwargs: captured.update(kwargs) or {'f1': 0.8}},
    )

    result = industrial._metric_evaluation_loop(
        target=np.array([[0], [1]]),
        predicted_labels=np.array([1, 0]),
        predicted_probs=np.array([[0.3, 0.7], [0.6, 0.4]]),
        problem='classification',
        metric_names=('f1',),
        rounding_order=3,
        train_data='train-data',
        seasonality=1,
    )

    assert result == {'f1': 0.8}
    assert np.array_equal(captured['target'], np.array([1, 2]))
    assert np.array_equal(captured['labels'], np.array([[2], [1]]))
    assert captured['metric_names'] == ('f1',)
    assert captured['train_data'] == 'train-data'
