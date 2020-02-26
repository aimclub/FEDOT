from abc import ABC, abstractmethod
from enum import Enum
from random import randint
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
class Composer(ABC):
    @abstractmethod
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      primary_requirements: List[Model],
                      secondary_requirements: List[Model],
                      metrics: Callable) -> Chain:
        raise NotImplementedError()


class DummyChainTypeEnum(Enum):
    flat = 1,
    hierarchical = 2


class DummyComposer(Composer):
    def __init__(self, dummy_chain_type):
        self.dummy_chain_type = dummy_chain_type

    # TODO move requirements to init
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      primary_requirements: List[Model],
                      secondary_requirements: List[Model],
                      metrics: Optional[Callable]) -> Chain:
        new_chain = Chain()

        if self.dummy_chain_type == DummyChainTypeEnum.hierarchical:
            # (y1, y2) -> y
            last_node = NodeGenerator.secondary_node(secondary_requirements[0])
            last_node.nodes_from = []

            for requirement_model in primary_requirements:
                new_node = NodeGenerator.primary_node(requirement_model, data)
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
        elif self.dummy_chain_type == DummyChainTypeEnum.flat:
            # (y1) -> (y2) -> y
            first_node = NodeGenerator.primary_node(primary_requirements[0], data)
            new_chain.add_node(first_node)
            prev_node = first_node
            for requirement_model in secondary_requirements:
                new_node = NodeGenerator.secondary_node(requirement_model)
                new_node.nodes_from = [prev_node]
                prev_node = new_node
                new_chain.add_node(new_node)
        else:
            raise NotImplementedError()
        return new_chain


class RandomSearchComposer:
    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      primary_requirements: List[Model],
                      secondary_requirements: List[Model],
                      metrics: Optional[Callable]) -> Chain:
        best_metric_value = 1000
        iter_num = 2000
        best_chain = None
        for i in range(iter_num):
            print(f'Iter {i}')
            new_chain = Chain()

            num_of_primary = randint(1, len(primary_requirements))
            num_of_secondary = randint(0, len(secondary_requirements))

            for _ in range(num_of_primary):
                random_first_model_ind = randint(0, len(primary_requirements) - 1)
                first_node = NodeGenerator.primary_node(primary_requirements[random_first_model_ind], data)
                new_chain.add_node(first_node)

            for _ in range(num_of_secondary):
                if randint(0, 1) == 1:
                    random_secondary_model_ind = randint(0, num_of_secondary - 1)
                    new_node = NodeGenerator.secondary_node(secondary_requirements[random_secondary_model_ind])
                    new_node.nodes_from = []
                    for _ in range(num_of_primary):
                        parent = randint(0, len(new_chain.nodes) - 1)
                        if parent != new_node:
                            new_node.nodes_from.append(new_chain.nodes[parent])
                    new_chain.add_node(new_node)
            new_metric_value = round(metrics(new_chain), 3)

            if len(new_chain.nodes) > 1:
                random_final_model_ind = randint(0, len(secondary_requirements) - 1)
                new_node = NodeGenerator.secondary_node(secondary_requirements[random_final_model_ind])
                new_node.nodes_from = [node for node in new_chain.nodes if node.nodes_from is None]
                if len(new_node.nodes_from) == 0:
                    new_node.nodes_from = new_chain.nodes
                new_chain.add_node(new_node)

            print(f'Try {new_metric_value} with length {new_chain.length} and depth {new_chain.depth}')
            if new_metric_value < best_metric_value:
                best_metric_value = new_metric_value
                best_chain = new_chain
                print(f'Better chain found: metric {best_metric_value}')

        return best_chain
