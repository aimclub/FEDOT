from dataclasses import dataclass
from typing import Optional, Sequence

from golem.core.optimisers.optimization_parameters import GraphRequirements

from fedot.core.pipelines.schemas import validate_cv_folds
from fedot.core.pipelines.pipeline_composer_requirements_rules import (
    PipelineEvaluationMode,
    resolve_pipeline_evaluation_mode,
)


@dataclass
class PipelineComposerRequirements(GraphRequirements):
    """Defines requirements on final Pipelines and data validation options.

    Restrictions on Pipelines:
    :param primary: available graph operation/content types
    :param secondary: available graph operation/content types

    Model validation options:
    :param cv_folds: number of cross-validation folds
    """

    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()
    cv_folds: Optional[int] = None
    evaluation_mode: PipelineEvaluationMode = PipelineEvaluationMode.default

    def __post_init__(self):
        super().__post_init__()
        validate_cv_folds(self.cv_folds)
        self.evaluation_mode = resolve_pipeline_evaluation_mode(self.evaluation_mode)
