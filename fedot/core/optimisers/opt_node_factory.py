from abc import abstractmethod, ABC
from random import choice
from typing import Optional

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode


class OptNodeFactory(ABC):
    @abstractmethod
    def exchange_node(self,
                      node: OptNode) -> Optional[OptNode]:
        """
        Returns new node based on a current node using information whether
        the node is primary or secondary and using advisor.

        :param node: current node that must be changed.
        """
        pass

    @abstractmethod
    def get_parent_node(self,
                        node: OptNode,
                        is_primary: bool) -> Optional[OptNode]:
        """
        Returns new parent node for the current node
        based on the content of the current node and using advisor.

        :param node: current node for which a parent node is generated
        :param is_primary: identifies whether to generate new parent node as separate primary node (if True)
        or to generate new intermediate secondary node between the current node and it's parent (if False)
        """
        pass

    @abstractmethod
    def get_node(self,
                 is_primary: bool) -> Optional[OptNode]:
        """
        Returns new child secondary node or new primary node for the current node
        based on the requirements for a node.
        """
        pass


class DefaultOptNodeFactory(OptNodeFactory):
    def __init__(self, requirements: Optional[PipelineComposerRequirements] = None):
        self.requirements = requirements

    def exchange_node(self, node: OptNode) -> Optional[OptNode]:
        return node

    def get_parent_node(self, node: OptNode, is_primary: bool) -> Optional[OptNode]:
        return self.get_node(is_primary=is_primary)

    def get_node(self, is_primary: bool) -> Optional[OptNode]:
        if not self.requirements:
            return None
        candidates = self.requirements.primary if is_primary else self.requirements.secondary
        return OptNode(content={'name': choice(candidates)})
