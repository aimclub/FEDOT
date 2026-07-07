from fedot.api.api_utils.api_service_rules import (
    build_tensordata_explain_plan,
    build_tensordata_fit_plan,
    build_tensordata_forecast_plan,
    build_tensordata_metrics_plan,
    build_tensordata_predict_plan,
    build_tensordata_predict_proba_plan,
    build_tensordata_tune_plan,
    build_tune_execution_plan_tensordata,
    resolve_forecast_horizon,
    resolve_predict_proba_mode,
)


def test_build_tune_execution_plan_uses_explicit_values_when_provided():
    plan = build_tune_execution_plan_tensordata(
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
    plan = build_tune_execution_plan_tensordata(
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


def test_service_rules_build_tensor_predict_execution_plans():
    predict_plan = build_tensordata_predict_plan(output_mode='labels')
    predict_proba_plan = build_tensordata_predict_proba_plan(
        probs_for_all_classes=True)

    assert predict_plan.output_mode == 'labels'
    assert predict_proba_plan.output_mode == 'full_probs'


def test_service_rules_build_tensordata_fit_plan_for_predefined_runtime_path():
    plan = build_tensordata_fit_plan('logit')

    assert plan.fit_method_name == 'fit_tensordata'


def test_service_rules_build_tensordata_tune_forecast_metrics_and_explain_plans():
    tune_plan = build_tensordata_tune_plan(
        converted_input_data='converted-input', has_tensor_data=True)
    legacy_tune_plan = build_tensordata_tune_plan(
        converted_input_data=None, has_tensor_data=False)
    forecast_plan = build_tensordata_forecast_plan(
        requested_horizon=None, forecast_length=12)
    metrics_plan = build_tensordata_metrics_plan()
    explain_plan = build_tensordata_explain_plan(
        method='surrogate_dt', visualization=False)

    assert tune_plan.input_data == 'converted-input'
    assert tune_plan.use_tensor_runtime is True
    assert tune_plan.builder_method_name == 'build_tensordata'
    assert tune_plan.refit_method_name == 'fit_tensordata'
    assert legacy_tune_plan.use_tensor_runtime is False
    assert legacy_tune_plan.builder_method_name == 'build'
    assert legacy_tune_plan.refit_method_name == 'fit'
    assert forecast_plan.horizon == 12
    assert forecast_plan.clear_target is True
    assert metrics_plan.output_mode == 'default'
    assert explain_plan.method == 'surrogate_dt'
    assert explain_plan.visualization is False


def test_service_rules_reject_unsupported_tensordata_fit_paths():
    import pytest

    with pytest.raises(ValueError, match='supports only predefined models or pipelines'):
        build_tensordata_fit_plan(None)

    with pytest.raises(ValueError, match='does not support auto assumption generation yet'):
        build_tensordata_fit_plan('auto')
