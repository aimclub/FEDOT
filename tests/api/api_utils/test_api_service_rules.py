from fedot.api.api_utils.api_service_rules import (
    build_explain_plan,
    build_forecast_plan,
    build_metrics_plan,
    build_predict_plan,
    build_predict_proba_plan,
    build_tune_execution_plan,
    resolve_forecast_horizon,
    resolve_predict_proba_mode,
)


def test_build_tune_execution_plan_uses_explicit_values_when_provided():
    plan = build_tune_execution_plan(
        tensor_data='new-data',
        train_data='train-data',
        requested_cv_folds=5,
        default_cv_folds=3,
        requested_n_jobs=2,
        default_n_jobs=1,
        requested_metric='roc_auc',
        default_metric='f1',
    )

    assert plan.tensor_data == 'new-data'
    assert plan.cv_folds == 5
    assert plan.n_jobs == 2
    assert plan.metric == 'roc_auc'


def test_build_tune_execution_plan_uses_defaults_when_values_are_missing():
    plan = build_tune_execution_plan(
        tensor_data=None,
        train_data='train-data',
        requested_cv_folds=None,
        default_cv_folds=3,
        requested_n_jobs=None,
        default_n_jobs=4,
        requested_metric=None,
        default_metric='f1',
    )

    assert plan.tensor_data == 'train-data'
    assert plan.cv_folds == 3
    assert plan.n_jobs == 4
    assert plan.metric == 'f1'


def test_service_rules_resolve_predict_mode_and_forecast_horizon():
    assert resolve_predict_proba_mode(False) == 'probs'
    assert resolve_predict_proba_mode(True) == 'full_probs'
    assert resolve_forecast_horizon(None, 12) == 12
    assert resolve_forecast_horizon(5, 12) == 5


def test_service_rules_build_tensor_predict_execution_plans():
    predict_plan = build_predict_plan(output_mode='labels')
    predict_proba_plan = build_predict_proba_plan(
        probs_for_all_classes=True)

    assert predict_plan.output_mode == 'labels'
    assert predict_proba_plan.output_mode == 'full_probs'


def test_service_rules_build_forecast_metrics_and_explain_plans():
    forecast_plan = build_forecast_plan(
        requested_horizon=None, forecast_length=12)
    metrics_plan = build_metrics_plan()
    explain_plan = build_explain_plan(
        method='surrogate_dt', visualization=False)

    assert forecast_plan.horizon == 12
    assert forecast_plan.clear_target is True
    assert metrics_plan.output_mode == 'default'
    assert explain_plan.method == 'surrogate_dt'
    assert explain_plan.visualization is False
