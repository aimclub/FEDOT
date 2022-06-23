from abc import abstractmethod

from testfixtures.compat import ABC

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.optimisers.graph import OptNode


class OptNodeFactory(ABC):
    @abstractmethod
    def change_node(self,
                    node: OptNode,
                    requirements,
                    advisor: DefaultChangeAdvisor):
        """
        Returns new node based on a current node using information weather
        the node is primary or secondary. Uses advisor.propose_change.

        :param node: current node that must be changed.
        :param requirements: requirements for changing the node
        :param advisor: DefaultChangeAdvisor to propose possible candidates for changing node
        """
        pass

    @abstractmethod
    def get_separate_parent_node(self,
                                 node: OptNode,
                                 requirements,
                                 advisor: DefaultChangeAdvisor):
        """
        Returns new parent node for the current node
        based on the content of the current node and using advisor.propose_parent.

        :param node: current node for which a parent node is generated
        :param requirements: requirements for generating new node
        :param advisor: DefaultChangeAdvisor to propose possible candidates for generating new parent node
        """
        pass

    @abstractmethod
    def get_intermediate_parent_node(self,
                                     node: OptNode,
                                     requirements,
                                     advisor: DefaultChangeAdvisor):
        """
        Returns new intermediate node between the current node and it's parent
        based on the content of the current node and the content of it's parent via adviser.propose_parent.

        :param node: current node for which the new node is generated
        :param requirements: requirements for generating new node
        :param advisor: DefaultChangeAdvisor to propose possible candidates for generating new node
        """
        pass

    @abstractmethod
    def get_child_node(self,
                       requirements):
        """
        Returns new child node for the current node
        based on the requirements for a secondary node.

        :param requirements: requirements for generating new node
        """
        pass

    @abstractmethod
    def get_primary_node(self,
                         requirements):
        """
        Returns new primary node according to requirements.

        :param requirements: requirements for generating new node
        """
        pass
