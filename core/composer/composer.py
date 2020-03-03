from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    List,
    Callable,
    Optional
)

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.model import Model


# TODO: specify ComposerRequirements class
class ComposerRequirements:
    def __init__(self, primary: List[Model], secondary: List[Model],
                 max_depth: Optional[int] = None,
                 max_arity: Optional[int] = None):
        self.primary = primary
        self.secondary = secondary
        self.max_depth = max_depth
        self.max_arity = max_arity


class Composer(ABC):
    @abstractmethod
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Callable) -> Chain:
        raise NotImplementedError()


class DummyChainTypeEnum(Enum):
    flat = 1,
    hierarchical = 2,
    hierarchical_2lev = 3


class DummyComposer(Composer):
    def __init__(self, dummy_chain_type):
        self.dummy_chain_type = dummy_chain_type

    # TODO move requirements to init
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Optional[Callable]) -> Chain:
        new_chain = Chain()

        if self.dummy_chain_type == DummyChainTypeEnum.hierarchical:
            # (y1, y2) -> y
            last_node = NodeGenerator.secondary_node(composer_requirements.secondary_requirements[0])
            last_node.nodes_from = []

            for requirement_model in composer_requirements.primary:
                new_node = NodeGenerator.primary_node(requirement_model, data)
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.hierarchical_2lev:
            # (((y1, y2) -> y3), ((y4, y5) -> y6)) - > y7
            last_node = NodeGenerator.secondary_node(composer_requirements.secondary[0])

            y1 = NodeGenerator.primary_node(composer_requirements.primary[0], data)
            new_chain.add_node(y1)

            y2 = NodeGenerator.primary_node(composer_requirements.primary[1], data)
            new_chain.add_node(y2)

            y3 = NodeGenerator.secondary_node(composer_requirements.secondary[1], [y1, y2])
            new_chain.add_node(y3)

            y4 = NodeGenerator.primary_node(composer_requirements.primary[2], data)
            new_chain.add_node(y4)
            y5 = NodeGenerator.primary_node(composer_requirements.primary[3], data)
            new_chain.add_node(y5)

            y6 = NodeGenerator.secondary_node(composer_requirements.secondary[4], [y4, y5])
            new_chain.add_node(y6)

            last_node.nodes_from = [y3, y6]

            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.flat:
            # (y1) -> (y2) -> y
            first_node = NodeGenerator.primary_node(composer_requirements.primary[0], data)
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
