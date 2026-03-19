import numpy as np
import pytest

from fedot import Fedot
from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _StubPipeline:
    def __init__(self):
        self.calls = []

    def predict(self, test_data, output_mode='default'):
        self.calls.append(output_mode)
        return OutputData(
            idx=np.arange(2),
            predict=np.array([[0.2, 0.8], [0.7, 0.3]]),
            target=None,
            task=Task(TaskTypesEnum.classification),
            data_type=DataTypesEnum.table,
        )


def test_main_facade_raises_not_fitted_errors_for_predictive_methods():
    model = Fedot(problem='classification')

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.predict(features=np.array([[1.0]]))

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.tune()

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.get_metrics()

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.return_report()


def test_main_facade_predict_proba_rejects_non_classification_tasks():
    model = Fedot(problem='regression')
    model.current_pipeline = object()

    with pytest.raises(ValueError, match='Probabilities of predictions are available only for classification'):
        model.predict_proba(features=np.array([[1.0]]))


def test_main_facade_uses_service_rule_for_predict_proba_mode_selection():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    model.target = 'target'
    model.data_processor.define_data = lambda **kwargs: type('Input',
                                                             (), {'task': Task(TaskTypesEnum.classification)})()

    model.predict_proba(features=np.array([[1.0], [2.0]]), probs_for_all_classes=True)

    assert model.current_pipeline.calls == ['full_probs']


def test_main_facade_forecast_requires_time_series_task():
    model = Fedot(problem='classification')
    model.current_pipeline = object()

    with pytest.raises(ValueError, match='Forecasting can be used only for the time series'):
        model.forecast()
