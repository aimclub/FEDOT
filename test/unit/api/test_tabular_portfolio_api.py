import numpy as np
import pytest

from fedot.api.experimental import tabular_portfolio
from fedot.api.experimental.tabular_portfolio import (
    TabularPortfolioResult,
    run_tabular_portfolio,
)


def _payload(predictions, probabilities=None):
    return {
        "predictions": np.asarray(predictions),
        "probabilities": probabilities,
        "training_duration": 1.25,
        "predict_duration": 0.25,
        "models_count": 2,
    }


def test_native_classification_api_encodes_train_labels_and_restores_predictions(
    monkeypatch,
):
    captured = {}

    def fake_run(dataset, config):
        captured.update(dataset=dataset, config=config)
        return _payload(
            [0, 1],
            probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        )

    monkeypatch.setattr(tabular_portfolio, "run", fake_run)
    truth = np.array(["no", "yes"])
    portfolio_result = run_tabular_portfolio(
        np.zeros((4, 2)),
        np.array(["yes", "no", "yes", "no"]),
        np.ones((2, 2)),
        task_type="classification",
        n_jobs=3,
        time_limit=180,
        test_target=truth,
        framework_params={"_portfolio": False},
    )

    assert isinstance(portfolio_result, TabularPortfolioResult)
    assert portfolio_result.predictions.tolist() == ["no", "yes"]
    assert portfolio_result.classes.tolist() == ["no", "yes"]
    assert portfolio_result.truth is truth
    assert captured["dataset"].train.y.tolist() == [1, 0, 1, 0]
    assert captured["dataset"].encoded_class_count == 2
    assert captured["config"].metric == "logloss"
    assert captured["config"].cores == 3
    assert captured["config"].max_runtime_seconds == 180
    assert captured["config"].framework_params["_portfolio"] is True


def test_native_regression_api_keeps_targets_and_uses_rmse(monkeypatch):
    captured = {}

    def fake_run(dataset, config):
        captured.update(dataset=dataset, config=config)
        return _payload([1.5, 2.5])

    monkeypatch.setattr(tabular_portfolio, "run", fake_run)
    target = np.array([0.5, 1.5, 2.5])
    portfolio_result = run_tabular_portfolio(
        np.zeros((3, 2)),
        target,
        np.ones((2, 2)),
        task_type="regression",
    )

    assert captured["dataset"].train.y is target
    assert captured["config"].metric == "rmse"
    assert portfolio_result.predictions.tolist() == [1.5, 2.5]
    assert portfolio_result.classes is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_type": "forecasting"}, "task_type"),
        ({"task_type": "classification", "time_limit": 0}, "time_limit"),
        ({"task_type": "classification", "n_jobs": 0}, "n_jobs"),
    ],
)
def test_native_portfolio_api_rejects_invalid_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_tabular_portfolio(
            np.zeros((4, 2)),
            np.array([0, 1, 0, 1]),
            np.ones((2, 2)),
            **kwargs,
        )
