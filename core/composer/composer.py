from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Callable,
    Optional
)

from core.composer.chain import Chain
from core.model import Model
from core.node import NodeGenerator


# TODO: specify ComposerRequirements class
class Composer(ABC):
    @abstractmethod
    def compose_chain(self, initial_chain: Optional[Chain], requirements: List[Any],
                      metrics: Callable):
        raise NotImplementedError()


class DummyComposer(Composer):
    def compose_chain(self, initial_chain: Optional[Chain], requirements: List[Model],
                      metrics: Callable) -> Chain:
        if initial_chain is None:
            new_chain = Chain()
            node_generator = NodeGenerator()
            last_node = node_generator.get_secondary_mode(requirements[0])
            last_node.nodes_from = []
            for requirement_model in requirements:
                new_node = node_generator.get_primary_mode(requirement_model)
                new_node.nodes_to = last_node
                new_chain.add_node(new_node)
                last_node.nodes_from.append(new_node)
            new_chain.add_node(last_node)
            return new_chain
        else:
            return initial_chain
