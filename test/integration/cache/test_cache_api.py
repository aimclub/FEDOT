import pytest

from fedot import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from test.data.datasets import get_dataset

# NOTE: check conftest.py
DEFAULT_TESTS_CACHE_API_PARAMS = {
    'timeout': 1,
    'preset': 'fast_train',
    'max_depth': 1,
    'max_arity': 2,
    'with_tuning': False,
    'use_stats': True,
}


@pytest.mark.parametrize('task_type, metric_name', [
    ('classification', 'f1'),
    ('regression', 'rmse'),
    ('ts_forecasting', 'rmse')
])
def test_cache_api(task_type, metric_name):
    if task_type == 'ts_forecasting':
        forecast_length = 1
        train_data, test_data, _ = get_dataset(task_type, validation_blocks=1,
                                               forecast_length=forecast_length)
        model = Fedot(
            problem=task_type,
            metric=metric_name,
            task_params=TsForecastingParams(forecast_length=forecast_length),
            **DEFAULT_TESTS_CACHE_API_PARAMS)
    else:
        train_data, test_data, _ = get_dataset(task_type, n_samples=100, n_features=10, iris_dataset=False)
        model = Fedot(problem=task_type, metric=metric_name, **DEFAULT_TESTS_CACHE_API_PARAMS)

    model.fit(features=train_data)
    model.predict(features=test_data)

    assert model.api_composer.predictions_cache is not None
    assert model.api_composer.predictions_cache._db.use_stats is not None
    assert model.api_composer.predictions_cache.effectiveness_ratio is not None
