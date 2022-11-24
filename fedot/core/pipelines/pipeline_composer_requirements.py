from dataclasses import dataclass
from typing import Optional, Sequence

from golem.core.optimisers.optimization_parameters import GraphRequirements


@dataclass
class PipelineComposerRequirements(GraphRequirements):
    """Defines requirements on final Pipelines and data validation options.

    Restrictions on Pipelines:
    :param primary: available graph operation/content types
    :param secondary: available graph operation/content types

    Model validation options:
    :param cv_folds: number of cross-validation folds
    :param validation_blocks: number of validation blocks for time series validation
    """

    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()

    cv_folds: Optional[int] = None
    validation_blocks: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError('Number of folds for KFold cross validation must be 2 or more.')
