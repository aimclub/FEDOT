from fedot.api.api_utils.api_service_rules import (
    build_tune_execution_plan,
    resolve_forecast_horizon,
    resolve_predict_proba_mode,
)


def test_build_tune_execution_plan_uses_explicit_values_when_provided():
    plan = build_tune_execution_plan(
        input_data='new-data',
        train_data='train-data',
        requested_cv_folds=5,
        default_cv_folds=3,
        requested_n_jobs=2,
        default_n_jobs=1,
        requested_metric='roc_auc',
        default_metric='f1',
    )

    assert plan.input_data == 'new-data'
    assert plan.cv_folds == 5
    assert plan.n_jobs == 2
    assert plan.metric == 'roc_auc'


def test_build_tune_execution_plan_uses_defaults_when_values_are_missing():
    plan = build_tune_execution_plan(
        input_data=None,
        train_data='train-data',
        requested_cv_folds=None,
        default_cv_folds=3,
        requested_n_jobs=None,
        default_n_jobs=4,
        requested_metric=None,
        default_metric='f1',
    )

    assert plan.input_data == 'train-data'
    assert plan.cv_folds == 3
    assert plan.n_jobs == 4
    assert plan.metric == 'f1'


def test_service_rules_resolve_predict_mode_and_forecast_horizon():
    assert resolve_predict_proba_mode(False) == 'probs'
    assert resolve_predict_proba_mode(True) == 'full_probs'
    assert resolve_forecast_horizon(None, 12) == 12
    assert resolve_forecast_horizon(5, 12) == 5
