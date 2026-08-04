from dataclasses import dataclass
from typing import Any, Union

from golem.utilities.data_structures import ComparableEnum as Enum

from fedot.core.repository.tasks import TaskTypesEnum


class OutputModeEnum(Enum):
    RAW = 'raw'
    AUTO = 'auto'
    DECODED = 'decoded'
    FLATTENED = 'flattened'


SUPPORTED_PIPELINE_OUTPUT_MODES = tuple(mode.value for mode in OutputModeEnum)


@dataclass(frozen=True)
class PipelinePostprocessPlan:
    should_flatten_prediction: bool
    should_restore_inverse_target_encoding: bool


def normalize_output_mode(output_mode: Union[OutputModeEnum, str, Any]) -> OutputModeEnum:
    """Normalize raw/enum output mode value to ``OutputModeEnum``."""
    if isinstance(output_mode, OutputModeEnum):
        return output_mode
    if hasattr(output_mode, 'value'):
        output_mode = output_mode.value
    return OutputModeEnum(output_mode)


def build_pipeline_postprocess_plan(
    output_mode: Union[OutputModeEnum, str],
    task_type: TaskTypesEnum,
) -> PipelinePostprocessPlan:
    from fedot.core.pipelines.schemas import validate_pipeline_output_mode

    mode = normalize_output_mode(validate_pipeline_output_mode(output_mode))

    if mode is OutputModeEnum.RAW:
        return PipelinePostprocessPlan(
            should_flatten_prediction=False,
            should_restore_inverse_target_encoding=False,
        )
    if mode is OutputModeEnum.AUTO:
        return PipelinePostprocessPlan(
            should_flatten_prediction=task_type is TaskTypesEnum.ts_forecasting,
            should_restore_inverse_target_encoding=task_type is TaskTypesEnum.classification,
        )
    if mode is OutputModeEnum.DECODED:
        return PipelinePostprocessPlan(
            should_flatten_prediction=False,
            should_restore_inverse_target_encoding=True,
        )
    return PipelinePostprocessPlan(
        should_flatten_prediction=True,
        should_restore_inverse_target_encoding=False,
    )
