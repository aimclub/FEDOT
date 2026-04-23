from dataclasses import dataclass

from fedot.core.data.tools import StateEnum
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class PipelinePreprocessPlan:
    should_preprocess: bool
    should_update_time_series_indices: bool


@dataclass(frozen=True)
class PipelinePostprocessPlan:
    should_restore_inverse_target_encoding: bool
    should_flatten_prediction: bool


@dataclass(frozen=True)
class PipelineTensorRuntimePlan:
    state: StateEnum
    index_method_name: str
    should_update_time_series_indices: bool


def build_pipeline_tensordata_runtime_plan(is_fit_stage: bool) -> PipelineTensorRuntimePlan:
    return PipelineTensorRuntimePlan(
        state=StateEnum.FIT if is_fit_stage else StateEnum.PREDICT,
        index_method_name='convert_indexes_for_fit' if is_fit_stage else 'convert_indexes_for_predict',
        should_update_time_series_indices=not is_fit_stage,
    )


def build_pipeline_preprocess_plan(is_fit_stage: bool, is_input_auto_preprocessed: bool) -> PipelinePreprocessPlan:
    return PipelinePreprocessPlan(
        should_preprocess=not is_input_auto_preprocessed,
        should_update_time_series_indices=not is_fit_stage,
    )


def build_pipeline_postprocess_plan(output_mode: str, task_type: TaskTypesEnum) -> PipelinePostprocessPlan:
    return PipelinePostprocessPlan(
        should_restore_inverse_target_encoding=output_mode == 'labels',
        should_flatten_prediction=task_type is TaskTypesEnum.ts_forecasting,
    )
