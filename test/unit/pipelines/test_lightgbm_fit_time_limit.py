from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.models import boostings_implementations
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotLightGBMClassificationImplementation,
    FedotLightGBMRegressionImplementation,
    _LightGBMFitTimeLimitCallback,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_lightgbm_fit_time_limit_stops_after_deadline(monkeypatch):
    clock = Mock(side_effect=[10.0, 14.9, 15.0])
    monkeypatch.setattr(boostings_implementations.time, 'monotonic', clock)
    callback = _LightGBMFitTimeLimitCallback(5)
    environment = SimpleNamespace(
        iteration=1,
        evaluation_result_list=[],
    )

    callback(environment)
    assert callback.stopped is False
    with pytest.raises(boostings_implementations.LGBMEarlyStopException):
        callback(environment)
    assert callback.stopped is True


@pytest.mark.parametrize('fit_time_limit', [0, -1, float('inf'), float('nan'), True, '5'])
def test_lightgbm_fit_time_limit_rejects_invalid_values(fit_time_limit):
    with pytest.raises(ValueError, match='positive finite number'):
        _LightGBMFitTimeLimitCallback(fit_time_limit)


@pytest.mark.parametrize(
    ('implementation_type', 'model_attribute'),
    [
        (FedotLightGBMClassificationImplementation, 'LGBMClassifier'),
        (FedotLightGBMRegressionImplementation, 'LGBMRegressor'),
    ],
)
def test_lightgbm_fit_time_limit_is_not_a_booster_param(
    monkeypatch, implementation_type, model_attribute
):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(boostings_implementations, model_attribute, constructor)

    implementation = implementation_type(
        OperationParameters(n_estimators=1_000, fit_time_limit=70)
    )

    assert 'fit_time_limit' not in constructor.call_args.kwargs
    callbacks = implementation.update_callbacks()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], _LightGBMFitTimeLimitCallback)
    assert callbacks[0].seconds == 70


@pytest.mark.parametrize(
    ('implementation_type', 'task_type', 'target'),
    [
        (
            FedotLightGBMClassificationImplementation,
            TaskTypesEnum.classification,
            np.arange(1_500) % 3,
        ),
        (
            FedotLightGBMRegressionImplementation,
            TaskTypesEnum.regression,
            np.linspace(-1, 1, 1_500),
        ),
    ],
)
def test_lightgbm_fit_time_limit_bounds_real_fit_without_eval_set(
    implementation_type, task_type, target
):
    random = np.random.RandomState(42)
    features = random.normal(size=(1_500, 20))
    implementation = implementation_type(
        OperationParameters(
            n_estimators=1_000,
            n_jobs=1,
            verbose=-1,
            use_eval_set=False,
            fit_time_limit=0.01,
        )
    )
    input_data = InputData(
        idx=np.arange(len(target)),
        features=features,
        target=target,
        task=Task(task_type),
        data_type=DataTypesEnum.table,
    )

    implementation.fit(input_data)

    assert 1 <= implementation.model.booster_.current_iteration() < 1_000
