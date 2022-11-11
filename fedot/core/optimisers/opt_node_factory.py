from abc import abstractmethod, ABC
from random import choice
from typing import Optional, Iterable

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode


class OptNodeFactory(ABC):
    @abstractmethod
    def exchange_node(self,
                      node: OptNode) -> Optional[OptNode]:
        """
        Returns new node based on a current node using information about node and advisor.

        :param node: current node that must be changed.
        """
        pass

    @abstractmethod
    def get_parent_node(self, node: OptNode, **kwargs) -> Optional[OptNode]:
        """
        Returns new parent node for the current node
        based on the content of the current node and using advisor.

        :param node: current node for which a parent node is generated
        """
        pass

    @abstractmethod
    def get_node(self, **kwargs) -> Optional[OptNode]:
        """
        Returns new node based on the requirements for a node.
        """
        pass


class DefaultOptNodeFactory(OptNodeFactory):
    def __init__(self, available_node_types: Iterable[str]):
        self._available_nodes = tuple(available_node_types)

    def exchange_node(self, node: OptNode) -> OptNode:
        return node

    def get_parent_node(self, node: OptNode, **kwargs) -> OptNode:
        return self.get_node(**kwargs)

    def get_node(self, **kwargs) -> OptNode:
        return OptNode(content={'name': choice(self._available_nodes)})
