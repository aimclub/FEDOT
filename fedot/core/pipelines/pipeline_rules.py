from dataclasses import dataclass
from typing import Any, Callable, Dict, Union

from golem.utilities.data_structures import ComparableEnum as Enum

from fedot.core.repository.tasks import TaskTypesEnum


class OutputModeEnum(Enum):
    """Pipeline predict output shaping modes.

    - ``RAW``: leave model output as-is (no flatten, no inverse target decode).
    - ``AUTO``: task-dependent defaults (decode for classification, flatten for
      TS forecasting).
    - ``DECODED``: restore original target labels via inverse target encoding.
    - ``FLATTENED``: ravel prediction to 1-D. Numeric-only (regression / TS);
      rejected for classification because it would keep encoded class ids.
    """

    RAW = 'raw'
    AUTO = 'auto'
    DECODED = 'decoded'
    FLATTENED = 'flattened'


SUPPORTED_PIPELINE_OUTPUT_MODES = tuple(mode.value for mode in OutputModeEnum)

# Legacy / operation-level modes still accepted by ``Pipeline.predict``.
SUPPORTED_OPERATION_OUTPUT_MODES = ('labels', 'probs', 'full_probs', 'default', False)


@dataclass(frozen=True)
class PipelinePostprocessPlan:
    should_flatten_prediction: bool
    should_restore_inverse_target_encoding: bool


@dataclass(frozen=True)
class PipelinePredictModes:
    """Split ``Pipeline.predict(output_mode=...)`` into node vs postprocess modes."""

    operation_output_mode: Any
    pipeline_output_mode: OutputModeEnum


def normalize_output_mode(output_mode: Union[OutputModeEnum, str, Any]) -> OutputModeEnum:
    """Normalize raw/enum output mode value to ``OutputModeEnum``."""
    if isinstance(output_mode, OutputModeEnum):
        return output_mode
    if hasattr(output_mode, 'value'):
        output_mode = output_mode.value
    return OutputModeEnum(output_mode)


_POSTPROCESS_PLAN_BY_MODE: Dict[
    OutputModeEnum,
    Callable[[TaskTypesEnum], PipelinePostprocessPlan],
] = {
    OutputModeEnum.RAW: lambda _task_type: PipelinePostprocessPlan(
        should_flatten_prediction=False,
        should_restore_inverse_target_encoding=False,
    ),
    OutputModeEnum.AUTO: lambda task_type: PipelinePostprocessPlan(
        should_flatten_prediction=task_type is TaskTypesEnum.ts_forecasting,
        should_restore_inverse_target_encoding=task_type is TaskTypesEnum.classification,
    ),
    OutputModeEnum.DECODED: lambda _task_type: PipelinePostprocessPlan(
        should_flatten_prediction=False,
        should_restore_inverse_target_encoding=True,
    ),
    OutputModeEnum.FLATTENED: lambda _task_type: PipelinePostprocessPlan(
        should_flatten_prediction=True,
        should_restore_inverse_target_encoding=False,
    ),
}


def resolve_pipeline_predict_modes(
    output_mode: Union[OutputModeEnum, str, Any, None],
    task_type: TaskTypesEnum,
) -> PipelinePredictModes:
    """
    Resolve ``Pipeline.predict`` output mode into operation + postprocess modes.

    Pipeline modes (``auto`` / ``raw`` / ``decoded`` / ``flattened``) drive
    ``_postprocess``. Operation modes (``labels`` / ``probs`` / …) are passed to
    the root node. Legacy operation modes keep postprocess as ``raw``.
    """
    if output_mode is None:
        output_mode = OutputModeEnum.AUTO

    mode_name = output_mode.value if hasattr(output_mode, 'value') else output_mode
    if mode_name in SUPPORTED_OPERATION_OUTPUT_MODES:
        return PipelinePredictModes(
            operation_output_mode=mode_name,
            pipeline_output_mode=OutputModeEnum.RAW,
        )

    from fedot.core.pipelines.schemas import validate_pipeline_output_mode

    pipeline_mode = normalize_output_mode(
        validate_pipeline_output_mode(output_mode, task_type=task_type)
    )
    needs_labels = (
        pipeline_mode in (OutputModeEnum.AUTO, OutputModeEnum.DECODED)
        and task_type is TaskTypesEnum.classification
    )
    return PipelinePredictModes(
        operation_output_mode='labels' if needs_labels else 'default',
        pipeline_output_mode=pipeline_mode,
    )


def build_pipeline_postprocess_plan(
    output_mode: Union[OutputModeEnum, str],
    task_type: TaskTypesEnum,
) -> PipelinePostprocessPlan:
    from fedot.core.pipelines.schemas import validate_pipeline_output_mode

    mode = normalize_output_mode(
        validate_pipeline_output_mode(output_mode, task_type=task_type)
    )
    return _POSTPROCESS_PLAN_BY_MODE[mode](task_type)
