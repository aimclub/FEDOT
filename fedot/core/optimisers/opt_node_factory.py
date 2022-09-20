from abc import abstractmethod, ABC
from random import choice
from typing import Optional, Any

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.optimisers.graph import OptNode
from fedot.core.repository.graph_model_reposiroty import GraphModelRepository
from fedot.core.utils import DEFAULT_PARAMS_STUB


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
    def __init__(self, requirements: Optional[Any] = None,
                 advisor: Optional[DefaultChangeAdvisor] = None,
                 graph_model_repository: Optional[GraphModelRepository] = None):
        self.requirements = requirements
        self.advisor = advisor or DefaultChangeAdvisor()
        self.graph_model_repository = graph_model_repository or self._init_default_graph_model_repo()

    def _init_default_graph_model_repo(self, **kwargs):
        """ Initialize default graph model repository with operations from composer requirements """
        pass

    def exchange_node(self, node: OptNode) -> Optional[OptNode]:
        return node

    def get_parent_node(self, node: OptNode, is_primary: bool) -> Optional[OptNode]:
        return self.get_node(is_primary=is_primary)

    def get_node(self, is_primary: bool) -> Optional[OptNode]:
        if not self.requirements:
            return None
        candidates = self.requirements.primary if is_primary else self.requirements.secondary
        return OptNode(content={'name': choice(candidates)})
