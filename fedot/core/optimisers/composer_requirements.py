import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from fedot.core.composer.advisor import PipelineChangeAdvisor


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :param primary: operation types for :class:`~fedot.core.pipelines.node.PrimaryNode`s
    :param secondary: operation types for :class:`~fedot.core.pipelines.node.SecondaryNode`s

    :param timeout: max time in minutes available for composition process
    :param stopping_after_n_generation: optional max number of stagnating populations for early stopping.
    Do not use early stopping if None.

    :param max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :param n_jobs: num of n_jobs
    :param show_progress: bool indicating whether to show progress using tqdm or not

    :param start_depth: start value of tree depth
    :param max_depth: max depth of the result pipeline
    :param max_arity: max number of parents for node
    :param min_arity: min number of parents for node

    :param cv_folds: number of cross-validation folds
    :param advisor: _description_
    """
    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()

    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    stopping_after_n_generation: Optional[int] = 10

    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    n_jobs: int = 1
    show_progress: bool = True

    start_depth: int = 3
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2

    cv_folds: Optional[int] = None
    advisor: Optional[PipelineChangeAdvisor] = field(default_factory=PipelineChangeAdvisor)

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError(f'Number of folds for KFold cross validation must be 2 or more.')
