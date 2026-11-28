import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from fedot.core.operations.evaluation.operation_implementations.models import boostings_implementations
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotXGBoostClassificationImplementation,
    FedotXGBoostRegressionImplementation,
    _FitTimeLimitCallback,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_xgboost_fit_time_limit_stops_after_deadline(monkeypatch):
    clock = Mock(side_effect=[10.0, 14.9, 15.0, 20.0, 24.9])
    monkeypatch.setattr(boostings_implementations.time, 'monotonic', clock)
    callback = _FitTimeLimitCallback(5)
    model = object()

    assert callback.before_training(model) is model
    assert not callback.before_iteration(model, epoch=0, evals_log={})
    assert not callback.before_iteration(model, epoch=1, evals_log={})
    assert callback.before_iteration(model, epoch=2, evals_log={})
    assert callback.after_training(model) is model
    assert callback.deadline is None
    assert callback.before_training(model) is model
    assert not callback.before_iteration(model, epoch=1, evals_log={})


def test_xgboost_fit_time_limit_adapts_eta_for_a_short_projected_horizon(monkeypatch):
    callback = _FitTimeLimitCallback(
        145,
        learning_rate=0.15,
        maximum_rounds=260,
        adaptive_learning_rate=True,
    )
    booster = Mock()
    iteration_starts = [0.0] + [float(epoch * 1.25) for epoch in range(17)]
    clock = Mock(side_effect=iteration_starts)
    monkeypatch.setattr(boostings_implementations.time, 'monotonic', clock)

    callback.before_training(booster)
    for epoch in range(17):
        assert not callback.before_iteration(booster, epoch=epoch, evals_log={})

    assert callback.projected_rounds == 102
    assert callback.adjusted_learning_rate == pytest.approx(0.2)
    booster.set_param.assert_called_once_with({'eta': pytest.approx(0.2)})


def test_xgboost_fit_time_limit_keeps_eta_when_full_horizon_is_reachable(monkeypatch):
    callback = _FitTimeLimitCallback(
        145,
        learning_rate=0.15,
        maximum_rounds=260,
        adaptive_learning_rate=True,
    )
    booster = Mock()
    iteration_starts = [0.0] + [float(epoch * 0.4) for epoch in range(17)]
    clock = Mock(side_effect=iteration_starts)
    monkeypatch.setattr(boostings_implementations.time, 'monotonic', clock)

    callback.before_training(booster)
    for epoch in range(17):
        assert not callback.before_iteration(booster, epoch=epoch, evals_log={})

    assert callback.projected_rounds == 260
    assert callback.adjusted_learning_rate is None
    booster.set_param.assert_not_called()


@pytest.mark.parametrize('fit_time_limit', [0, -1, float('inf'), float('nan'), True, '5'])
def test_xgboost_fit_time_limit_rejects_invalid_values(fit_time_limit):
    with pytest.raises(ValueError, match='positive finite number'):
        _FitTimeLimitCallback(fit_time_limit)


@pytest.mark.parametrize(
    ('implementation_type', 'model_attribute'),
    [
        (FedotXGBoostClassificationImplementation, 'XGBClassifier'),
        (FedotXGBoostRegressionImplementation, 'XGBRegressor'),
    ],
)
def test_xgboost_fit_time_limit_is_converted_to_constructor_callback(
    monkeypatch, implementation_type, model_attribute
):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(boostings_implementations, model_attribute, constructor)

    implementation_type(
        OperationParameters(
            n_estimators=260,
            verbosity=0,
            fit_time_limit=145,
        )
    )

    model_params = constructor.call_args.kwargs
    assert 'fit_time_limit' not in model_params
    assert model_params['n_estimators'] == 260
    assert len(model_params['callbacks']) == 1
    assert isinstance(model_params['callbacks'][0], _FitTimeLimitCallback)
    assert model_params['callbacks'][0].seconds == 145


def test_xgboost_deadline_adaptive_learning_rate_is_opt_in_and_not_a_booster_param(monkeypatch):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(boostings_implementations, 'XGBClassifier', constructor)

    FedotXGBoostClassificationImplementation(
        OperationParameters(
            n_estimators=260,
            learning_rate=0.15,
            verbosity=0,
            fit_time_limit=145,
            fit_time_limit_adaptive_learning_rate=True,
        )
    )

    model_params = constructor.call_args.kwargs
    callback = model_params['callbacks'][0]
    assert 'fit_time_limit_adaptive_learning_rate' not in model_params
    assert callback.adaptive_learning_rate is True
    assert callback.learning_rate == pytest.approx(0.15)
    assert callback.maximum_rounds == 260


@pytest.mark.parametrize(
    ('implementation_type', 'task_type', 'target'),
    [
        (
            FedotXGBoostClassificationImplementation,
            TaskTypesEnum.classification,
            np.arange(1_500) % 3,
        ),
        (
            FedotXGBoostRegressionImplementation,
            TaskTypesEnum.regression,
            np.linspace(-1, 1, 1_500),
        ),
    ],
)
def test_xgboost_fit_time_limit_bounds_real_repeated_fit_and_is_serializable(
    implementation_type, task_type, target
):
    random = np.random.RandomState(42)
    features = random.normal(size=(1_500, 20))
    implementation = implementation_type(
        OperationParameters(
            n_estimators=1_000,
            max_depth=6,
            n_jobs=1,
            verbosity=0,
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
    first_rounds = implementation.model.get_booster().num_boosted_rounds()
    restored_model = pickle.loads(pickle.dumps(implementation.model))
    predictions = restored_model.predict(features[:5])
    implementation.fit(input_data)
    second_rounds = implementation.model.get_booster().num_boosted_rounds()

    assert 1 <= first_rounds < 1_000
    assert 1 <= second_rounds < 1_000
    assert predictions.shape == (5,)
