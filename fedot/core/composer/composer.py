import datetime
from abc import ABC, abstractmethod
from typing import (Any, Callable, List, Optional)

from dataclasses import dataclass

from fedot.core.composer.chain import Chain
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.log import Log, default_log
from fedot.core.models.data import InputData
from fedot.core.utils import ComparableEnum as Enum


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :param primary: List of model types (str) for Primary Nodes
    :param secondary: List of model types (str) for Secondary Nodes
    :param max_lead_time: max time in minutes available for composition process
    :param max_depth: max depth of the result chain
    :param max_arity: maximal number of parent for node
    :param min_arity: minimal number of parent for node
    :param add_single_model_chains: allow to have chain with only one node
    """
    primary: List[str]
    secondary: List[str]
    max_lead_time: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2
    add_single_model_chains: bool = True

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')


class Composer(ABC):
    """
    Base class used for receiving composite models via optimization
    :param metrics: metrics used to define the quality of found solution
    :param composer_requirements: requirements for composition process
    :param optimiser_parameters: parameters used by optimization process (i.e. GPComposerRequirements)
    :param initial_chain: defines the initial state of the population. If None then initial population is random.
    :param log: optional parameter for log oject
    """

    def __init__(self, metrics: Optional[Callable], composer_requirements: ComposerRequirements,
                 optimiser_parameters: Any = None, initial_chain: Optional[Chain] = None,
                 log: Log = None):
        self.history = None
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


class DummyChainTypeEnum(Enum):
    flat = 1,
    hierarchical = 2


class DummyComposer(Composer):
    def __init__(self, dummy_chain_type: DummyChainTypeEnum,
                 composer_requirements: ComposerRequirements,
                 metrics: Optional[Callable],
                 initial_chain: Optional[Chain] = None):
        super().__init__(composer_requirements=composer_requirements, metrics=metrics,
                         initial_chain=initial_chain)
        self.dummy_chain_type = dummy_chain_type

    def compose_chain(self, data: InputData, is_visualise: bool = False) -> Chain:
        new_chain = Chain()

        if self.dummy_chain_type == DummyChainTypeEnum.hierarchical:
            # (y1, y2) -> y
            last_node = SecondaryNode(self.composer_requirements.secondary[0])

            for requirement_model in self.composer_requirements.primary:
                new_node = PrimaryNode(requirement_model)
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.flat:
            # (y1) -> (y2) -> y
            first_node = PrimaryNode(self.composer_requirements.primary[0])
            new_chain.add_node(first_node)
            prev_node = first_node
            for requirement_model in self.composer_requirements.secondary:
                new_node = SecondaryNode(requirement_model)
                new_node.nodes_from = [prev_node]
                prev_node = new_node
                new_chain.add_node(new_node)
        else:
            raise NotImplementedError()
        return new_chain
