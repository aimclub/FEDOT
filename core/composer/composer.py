from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    List,
    Callable,
    Optional
)

import numpy as np

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import Data
from core.models.model import Model


# TODO: specify ComposerRequirements class
class ComposerRequirements:
    def __init__(self, primary_requirements: List[Model], secondary_requirements: List[Model]):
        self.primary_requirements= primary_requirements
        self.secondary_requirements =secondary_requirements

class Composer(ABC):
    @abstractmethod
    def compose_chain(self,data: Optional[Data], initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Callable) -> Chain:
        raise NotImplementedError()


class DummyChainTypeEnum(Enum):
    flat = 1,
    hierarchical = 2


class DummyComposer(Composer):
    def __init__(self, dummy_chain_type):
        self.dummy_chain_type = dummy_chain_type

    # TODO move requirements to init
    def compose_chain(self, data: Optional[Data],
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Optional[Callable]) -> Chain:
        new_chain = Chain()
        if not data:
            data = Data(np.zeros(1), np.zeros(1), np.zeros(1)) #empty data

        if self.dummy_chain_type == DummyChainTypeEnum.hierarchical:
            # (y1, y2) -> y
            last_node = NodeGenerator.get_secondary_node(composer_requirements.secondary_requirements[0])
            last_node.nodes_from = []

            for requirement_model in composer_requirements.primary_requirements:
                new_node = NodeGenerator.get_primary_node(requirement_model, data)
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.flat:
            # (y1) -> (y2) -> y
            first_node = NodeGenerator.get_primary_node(composer_requirements.primary_requirements[0], data)
            new_chain.add_node(first_node)
            prev_node = first_node
            for requirement_model in composer_requirements.secondary_requirements:
                new_node = NodeGenerator.get_secondary_node(requirement_model)
                new_node.nodes_from = [prev_node]
                prev_node = new_node
                new_chain.add_node(new_node)
        else:
            raise NotImplemented
        return new_chain
