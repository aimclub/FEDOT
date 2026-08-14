from enum import Enum
from typing import Union


class PipelineEvaluationMode(Enum):
    default = 'default'
    oof = 'oof'


def resolve_pipeline_evaluation_mode(
    mode: Union[PipelineEvaluationMode, str],
) -> PipelineEvaluationMode:
    """Normalize the public pipeline evaluation mode to its domain enum."""
    if isinstance(mode, PipelineEvaluationMode):
        return mode
    try:
        return PipelineEvaluationMode(mode)
    except ValueError as ex:
        supported = ', '.join(item.value for item in PipelineEvaluationMode)
        raise ValueError(
            f'Unsupported pipeline evaluation mode {mode!r}. Expected one of: {supported}'
        ) from ex
