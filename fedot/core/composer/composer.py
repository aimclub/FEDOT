from abc import ABC, abstractmethod
from typing import List, Optional, Union, Sequence
from dataclasses import dataclass
import datetime

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.optimisers.optimizer import GraphOptimizer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.composer.advisor import PipelineChangeAdvisor


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    Args:
        primary: operation types for :class:`PrimaryNode`
        secondary: operation types for :class:`SecondaryNode`
        timeout: max time in minutes available for composition process
        max_pipeline_fit_time: time constraint for operation fitting (minutes)
        max_depth: max depth of the result pipeline
        max_arity: max number of parents for node
        min_arity: min number of parents for node
        cv_folds: number of cross-validation folds
        advisor: :obj:`PipelineChangeAdvisor` object
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


class Composer(ABC):
    """
    Base class used for receiving composite operations via optimization

    Args:
        optimiser: optimiser generated in :class:`~fedot.core.composer.ComposerBuilder`
        composer_requirements: requirements for composition process
        initial_pipelines: defines the initial state of the population. If None then initial population is random.
    """

    def __init__(self, optimizer: GraphOptimizer, composer_requirements: Optional[ComposerRequirements] = None):
        self.composer_requirements = composer_requirements
        self.optimizer = optimizer
        self.log = default_log(self)

    @abstractmethod
    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, List[Pipeline]]:
        """
        Run composition process for optimal pipeline structure search

        Args:
            data: Data used for problem solving

        Returns:
            For ``single-objective optimization`` -> the best pipeline.\n
            For ``multi-objective optimization`` -> a list of the best pipelines is returned.

        Notes:
            Returned pipelines are ordered by the descending primary metric (the first is the best).
        """

        raise NotImplementedError()
