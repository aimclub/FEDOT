import datetime
from dataclasses import dataclass
from typing import Sequence, Optional

from fedot.core.composer.advisor import PipelineChangeAdvisor


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :attribute primary: List of operation types (str) for Primary Nodes
    :attribute secondary: List of operation types (str) for Secondary Nodes
    :attribute timeout: max time in minutes available for composition process
    :attribute max_depth: max depth of the result pipeline
    :attribute max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :attribute max_arity: maximal number of parent for node
    :attribute min_arity: minimal number of parent for node
    :attribute cv_folds: integer or None to use cross validation
    """
    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2
    cv_folds: Optional[int] = None
    advisor: Optional[PipelineChangeAdvisor] = PipelineChangeAdvisor()

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError(f'Number of folds for KFold cross validation must be 2 or more.')
