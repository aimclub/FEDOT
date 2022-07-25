from abc import abstractmethod, ABC
from random import choice
from typing import Optional, Any

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.optimisers.graph import OptNode
from fedot.core.utils import DEFAULT_PARAMS_STUB


class OptNodeFactory(ABC):
    def __init__(self, requirements: Optional[Any] = None,
                 advisor: Optional[DefaultChangeAdvisor] = None):
        self.requirements = requirements
        self.advisor = advisor or DefaultChangeAdvisor()

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
                        primary: bool) -> Optional[OptNode]:
        """
        Returns new parent node for the current node
        based on the content of the current node and using advisor.

        :param node: current node for which a parent node is generated
        :param primary: identifies whether to generate new parent node as separate primary node (if True)
        or to generate new intermediate secondary node between the current node and it's parent (if False)
        """
        pass

    @abstractmethod
    def get_node(self,
                 primary: bool) -> Optional[OptNode]:
        """
        Returns new child secondary node or new primary node for the current node
        based on the requirements for a node.
        """
        pass


class DefaultOptNodeFactory(OptNodeFactory):
    def __init__(self, requirements: Optional[Any] = None,
                 advisor: Optional[DefaultChangeAdvisor] = None):
        super().__init__(requirements, advisor)

    def exchange_node(self, node: OptNode) -> Optional[OptNode]:
        return node

    def get_parent_node(self, node: OptNode, primary: bool) -> Optional[OptNode]:
        return self.get_node(primary=primary)

    def get_node(self, primary: bool) -> Optional[OptNode]:
        if self.requirements:
            candidates = self.requirements.primary if primary else self.requirements.secondary
            return OptNode(content={'name': choice(candidates)})
        else:
            return None
