from fedot.core.pipelines.pipeline_rules import build_pipeline_postprocess_plan
from fedot.core.repository.tasks import TaskTypesEnum


def test_build_pipeline_postprocess_plan_handles_ts_flatten():
    labels_plan = build_pipeline_postprocess_plan(
        'labels', TaskTypesEnum.classification)
    ts_plan = build_pipeline_postprocess_plan(
        'default', TaskTypesEnum.ts_forecasting)

    assert labels_plan.should_flatten_prediction is False
    assert ts_plan.should_flatten_prediction is True
