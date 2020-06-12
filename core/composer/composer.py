import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, List, Optional)

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData


@dataclass
class ComposerRequirements:
    primary: List[str]
    secondary: List[str]
    max_lead_time: Optional[datetime.timedelta] = datetime.timedelta(minutes=30)
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
    def __init__(self):
        self.history = None

    @abstractmethod
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Callable,
                      optimiser_parameters: Any = None,
                      is_visualise: bool = False) -> Chain:
        raise NotImplementedError()


class DummyChainTypeEnum(Enum):
    flat = 1,
    hierarchical = 2


class DummyComposer(Composer):
    def __init__(self, dummy_chain_type):
        super(Composer, self).__init__()
        self.dummy_chain_type = dummy_chain_type

    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Optional[Callable],
                      optimiser_parameters=None,
                      is_visualise: bool = False) -> Chain:
        new_chain = Chain()

        if self.dummy_chain_type == DummyChainTypeEnum.hierarchical:
            # (y1, y2) -> y
            last_node = NodeGenerator.secondary_node(composer_requirements.secondary[0])

            for requirement_model in composer_requirements.primary:
                new_node = NodeGenerator.primary_node(requirement_model)
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.flat:
            # (y1) -> (y2) -> y
            first_node = NodeGenerator.primary_node(composer_requirements.primary[0])
            new_chain.add_node(first_node)
            prev_node = first_node
            for requirement_model in composer_requirements.secondary:
                new_node = NodeGenerator.secondary_node(requirement_model)
                new_node.nodes_from = [prev_node]
                prev_node = new_node
                new_chain.add_node(new_node)
        else:
            raise NotImplementedError()
        return new_chain
