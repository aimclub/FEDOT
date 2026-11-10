from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.models import boostings_implementations
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotCatBoostClassificationImplementation,
    FedotCatBoostRegressionImplementation,
    _CatBoostFitTimeLimitCallback,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_catboost_fit_time_limit_stops_after_deadline_and_resets(monkeypatch):
    clock = Mock(side_effect=[10.0, 14.9, 15.0, 30.0, 31.0])
    monkeypatch.setattr(boostings_implementations.time, 'monotonic', clock)
    callback = _CatBoostFitTimeLimitCallback(5)

    assert callback.after_iteration(SimpleNamespace(iteration=1)) is True
    assert callback.after_iteration(SimpleNamespace(iteration=2)) is True
    assert callback.after_iteration(SimpleNamespace(iteration=3)) is False
    assert callback.stopped is True
    assert callback.after_iteration(SimpleNamespace(iteration=1)) is True
    assert callback.after_iteration(SimpleNamespace(iteration=2)) is True
    assert callback.stopped is False


@pytest.mark.parametrize(
    'fit_time_limit', [0, -1, float('inf'), float('nan'), True, '5']
)
def test_catboost_fit_time_limit_rejects_invalid_values(fit_time_limit):
    with pytest.raises(ValueError, match='positive finite number'):
        _CatBoostFitTimeLimitCallback(fit_time_limit)


@pytest.mark.parametrize(
    ('implementation_type', 'model_attribute'),
    [
        (FedotCatBoostClassificationImplementation, 'CatBoostClassifier'),
        (FedotCatBoostRegressionImplementation, 'CatBoostRegressor'),
    ],
)
def test_catboost_fit_time_limit_is_a_fit_callback_not_a_model_param(
        monkeypatch, implementation_type, model_attribute):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(boostings_implementations, model_attribute, constructor)

    implementation = implementation_type(
        OperationParameters(n_jobs=2, iterations=1_000, fit_time_limit=70)
    )

    assert 'fit_time_limit' not in constructor.call_args.kwargs
    assert 'callbacks' not in constructor.call_args.kwargs
    assert len(implementation.fit_callbacks) == 1
    assert isinstance(
        implementation.fit_callbacks[0], _CatBoostFitTimeLimitCallback
    )
    assert implementation.fit_callbacks[0].seconds == 70


@pytest.mark.parametrize(
    ('implementation_type', 'task_type', 'target'),
    [
        (
            FedotCatBoostClassificationImplementation,
            TaskTypesEnum.classification,
            np.arange(1_500) % 3,
        ),
        (
            FedotCatBoostRegressionImplementation,
            TaskTypesEnum.regression,
            np.linspace(-1, 1, 1_500),
        ),
    ],
)
def test_catboost_fit_time_limit_bounds_real_fit(
        implementation_type, task_type, target):
    random = np.random.RandomState(42)
    features = random.normal(size=(1_500, 20))
    implementation = implementation_type(
        OperationParameters(
            iterations=1_000,
            depth=6,
            n_jobs=1,
            verbose=False,
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

    assert 1 <= implementation.model.tree_count_ < 1_000
