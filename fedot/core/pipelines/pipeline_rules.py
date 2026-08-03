from dataclasses import dataclass

from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class PipelinePostprocessPlan:
    should_flatten_prediction: bool


def build_pipeline_postprocess_plan(output_mode: str, task_type: TaskTypesEnum) -> PipelinePostprocessPlan:
    return PipelinePostprocessPlan(
        should_flatten_prediction=task_type is TaskTypesEnum.ts_forecasting,
    )
