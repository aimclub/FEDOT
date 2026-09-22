import numpy as np

from fedot.core.data.input_data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.api.main_rules import (
    build_industrial_explain_plan,
    build_industrial_finetune_plan,
    build_industrial_fit_plan,
    build_industrial_history_visualization_plan,
    build_industrial_load_plan,
    build_industrial_metrics_plan,
    build_industrial_metrics_request_plan,
    build_industrial_predict_plan,
    build_industrial_predict_proba_plan,
    build_industrial_save_plan,
    normalize_industrial_prediction,
    trim_industrial_forecast,
)
from fedot.industrial.core.repository.constanst_repository import FEDOT_TUNER_STRATEGY, FEDOT_TUNING_METRICS


def test_build_industrial_predict_plan_tracks_solver_mode_and_forecast_tail():
    plan = build_industrial_predict_plan(
        predict_mode='labels',
        solver_is_fedot_class=False,
        solver_is_pipeline_class=True,
        has_target_encoder=True,
        predict_task=Task(TaskTypesEnum.ts_forecasting,
                          TsForecastingParams(forecast_length=3)),
    )

    assert plan.custom_predict is False
    assert plan.labels_output is True
    assert plan.use_pipeline_predict_mode is True
    assert plan.forecast_length == 3


def test_build_industrial_metrics_plan_detects_mapping_and_encoder_usage():
    dict_plan = build_industrial_metrics_plan(
        np.array([[1], [0]]), {'a': np.array([1, 0])}, True)
    array_plan = build_industrial_metrics_plan(
        np.array([[1], [0]]), np.array([1, 0]), True)

    assert dict_plan.prediction_is_mapping is True
    assert dict_plan.use_target_encoder is False
    assert array_plan.prediction_is_mapping is False
    assert array_plan.use_target_encoder is True


def test_normalize_industrial_prediction_unwraps_outputdata_and_trim_forecast_is_optional():
    raw = OutputData(
        idx=np.arange(4),
        predict=np.array([1, 2, 3, 4]),
        target=None,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    normalized = normalize_industrial_prediction(raw)

    assert np.array_equal(normalized, np.array([1, 2, 3, 4]))
    assert np.array_equal(trim_industrial_forecast(
        normalized, None), normalized)
    assert np.array_equal(trim_industrial_forecast(
        normalized, 2), np.array([3, 4]))


def test_build_industrial_fit_predict_proba_and_metrics_request_plans():
    class _CallableStrategy:
        def __call__(self):
            return None

    fit_plan = build_industrial_fit_plan(object())
    callable_fit_plan = build_industrial_fit_plan(_CallableStrategy())
    proba_plan = build_industrial_predict_proba_plan(
        'probs', is_regression_task_context=True)
    metrics_request_plan = build_industrial_metrics_request_plan(
        problem='classification',
        probs=None,
        metric_names=('roc_auc',),
    )

    assert fit_plan.use_solver_fit is True
    assert callable_fit_plan.use_solver_fit is False
    assert proba_plan.normalized_mode == 'labels'
    assert metrics_request_plan.warn_missing_probabilities is True


def test_build_industrial_finetune_plan_normalizes_metric_and_tuner():
    plan = build_industrial_finetune_plan(
        is_fedot_datatype=False,
        task_name='classification',
        tuning_params={'tuner': 'sequential', 'iterations': 5},
    )

    assert plan.should_process_input is True
    assert plan.normalized_tuning_params['iterations'] == 5
    assert plan.normalized_tuning_params['metric'] == FEDOT_TUNING_METRICS['classification']
    assert plan.normalized_tuning_params['tuner'] == FEDOT_TUNER_STRATEGY['sequential']


def test_build_industrial_save_load_explain_and_history_plans():
    save_plan = build_industrial_save_plan(mode='all', is_fedot_solver=False)
    load_plan = build_industrial_load_plan(
        path='root', dir_list=['pipeline_saved_1', 'fitted_operations'])
    explain_plan = build_industrial_explain_plan(
        {'method': 'recurrence', 'samples': 3})
    history_plan = build_industrial_history_visualization_plan('models')

    assert save_plan.save_all is True
    assert save_plan.include_opt_hist is False
    assert load_plan.resolved_path == 'root/pipeline_saved_1'
    assert load_plan.load_multiple_pipelines is True
    assert explain_plan.method == 'recurrence'
    assert explain_plan.samples == 3
    assert explain_plan.metric == 'rmse'
    assert history_plan.selected_mode == 'models'
    assert history_plan.visualize_all is False
