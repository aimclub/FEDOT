import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Any, Union, List, Optional)

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.quality_metrics_repository import (MetricsEnum)


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :attribute primary: List of operation types (str) for Primary Nodes
    :attribute secondary: List of operation types (str) for Secondary Nodes
    :attribute max_lead_time: max time in minutes available for composition process
    :attribute max_depth: max depth of the result chain
    :attribute max_chain_fit_time: time constraint for operation fitting (minutes)
    :attribute max_arity: maximal number of parent for node
    :attribute min_arity: minimal number of parent for node
    :attribute allow_single_operations: allow to have chain with only one node
    """
    primary: List[str]
    secondary: List[str]
    max_lead_time: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    max_chain_fit_time: Optional[datetime.timedelta] = None
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2
    allow_single_operations: bool = False

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')


class Composer(ABC):
    """
    Base class used for receiving composite operations via optimization
    :param metrics: metrics used to define the quality of found solution
    :param composer_requirements: requirements for composition process
    :param optimiser_parameters: parameters used by optimization process (i.e. GPComposerRequirements)
    :param initial_chain: defines the initial state of the population. If None then initial population is random.
    :param log: optional parameter for log oject
    """

    def __init__(self, metrics: Union[List[MetricsEnum], MetricsEnum], composer_requirements: ComposerRequirements,
                 optimiser_parameters: Any = None, initial_chain: Optional[Chain] = None,
                 log: Log = None):
        self.metrics = metrics
        self.composer_requirements = composer_requirements
        self.optimiser_parameters = optimiser_parameters
        self.initial_chain = initial_chain

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def compose_chain(self, data: InputData,
                      is_visualise: bool = False) -> Chain:
        """
        Base method to run the composition process

        :param data: data used for problem solving
        :param is_visualise: flag to enable visualization. Default False.
        :return: Chain object
        """
        raise NotImplementedError()
