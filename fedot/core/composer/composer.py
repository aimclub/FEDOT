import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (List, Optional, Union, Sequence)

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.optimisers.optimizer import GraphOptimiser
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsEnum


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
    primary: List[str]
    secondary: List[str]
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
    :param optimiser: optimiser generated in ComposerBuilder
    :param composer_requirements: requirements for composition process
    :param initial_pipelines: defines the initial state of the population. If None then initial population is random.
    """

    def __init__(self, optimiser: GraphOptimiser,
                 composer_requirements: ComposerRequirements,
                 initial_pipelines: Optional[Sequence[Pipeline]] = None):
        self.composer_requirements = composer_requirements
        self.initial_pipelines = initial_pipelines
        self.optimiser = optimiser
        self.log = default_log(self.__class__.__name__)

    @abstractmethod
    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, List[Pipeline]]:
        """ Run composition process for optimal pipeline structure search.
        :param data: Data used for problem solving.
        :return: Best composed pipeline or pipelines.
         For single-objective optimization -- the best pipeline.
         For multi-objective optimization -- a list of the best pipelines is returned.
         Returned pipelines are ordered by the descending primary metric (the first is the best).
        """

        raise NotImplementedError()
