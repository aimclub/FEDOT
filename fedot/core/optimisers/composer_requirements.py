import datetime
from dataclasses import dataclass
from typing import Optional, Sequence

from fedot.core.composer.advisor import PipelineChangeAdvisor


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :param primary: operation types for :class:`~fedot.core.pipelines.node.PrimaryNode`s
    :param secondary: operation types for :class:`~fedot.core.pipelines.node.SecondaryNode`s
    :param timeout: max time in minutes available for composition process
    :param max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :param max_depth: max depth of the result pipeline
    :param max_arity: max number of parents for node
    :param min_arity: min number of parents for node
    :param cv_folds: number of cross-validation folds
    :param advisor: _description_
    """
    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2
    cv_folds: Optional[int] = None
    advisor: Optional[PipelineChangeAdvisor] = PipelineChangeAdvisor()  # TODO: use factory to mutable object!

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError(f'Number of folds for KFold cross validation must be 2 or more.')
