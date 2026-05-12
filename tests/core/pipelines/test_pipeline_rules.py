from fedot.core.data.common.enums import StateEnum
from fedot.core.pipelines.pipeline_rules import (
    build_pipeline_postprocess_plan,
    build_pipeline_preprocess_plan,
    build_pipeline_tensordata_runtime_plan,
)
from fedot.core.repository.tasks import TaskTypesEnum


def test_build_pipeline_preprocess_plan_handles_fit_and_predict_stages():
    fit_plan = build_pipeline_preprocess_plan(
        is_fit_stage=True, is_input_auto_preprocessed=False)
    predict_plan = build_pipeline_preprocess_plan(
        is_fit_stage=False, is_input_auto_preprocessed=True)

    assert fit_plan.should_preprocess is True
    assert fit_plan.should_update_time_series_indices is False
    assert predict_plan.should_preprocess is False
    assert predict_plan.should_update_time_series_indices is True


def test_build_pipeline_postprocess_plan_handles_labels_and_ts_outputs():
    labels_plan = build_pipeline_postprocess_plan(
        'labels', TaskTypesEnum.classification)
    ts_plan = build_pipeline_postprocess_plan(
        'default', TaskTypesEnum.ts_forecasting)

    assert labels_plan.should_restore_inverse_target_encoding is True
    assert labels_plan.should_flatten_prediction is False
    assert ts_plan.should_restore_inverse_target_encoding is False
    assert ts_plan.should_flatten_prediction is True


def test_build_pipeline_tensordata_runtime_plan_selects_stage_specific_behavior():
    fit_plan = build_pipeline_tensordata_runtime_plan(is_fit_stage=True)
    predict_plan = build_pipeline_tensordata_runtime_plan(is_fit_stage=False)

    assert fit_plan.state == StateEnum.FIT
    assert fit_plan.index_method_name == 'convert_indexes_for_fit'
    assert fit_plan.should_update_time_series_indices is False
    assert predict_plan.state == StateEnum.PREDICT
    assert predict_plan.index_method_name == 'convert_indexes_for_predict'
    assert predict_plan.should_update_time_series_indices is True
