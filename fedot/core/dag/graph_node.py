from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Hashable

from fedot.core.dag.graph_utils import node_depth, descriptive_id_recursive


class GraphNode(ABC, Hashable):
    """Definition of the node in directed graph structure.

    Provides interface for getting and modifying the parent nodes
    and recursive description based on all preceding nodes.
    """

    @property
    @abstractmethod
    def nodes_from(self) -> List['GraphNode']:
        """Gets all parent nodes of this graph node

        Returns:
            List['GraphNode']: all the parent nodes
        """
        pass

    @nodes_from.setter
    @abstractmethod
    def nodes_from(self, nodes: Optional[Iterable['GraphNode']]):
        """Changes value of parent nodes of this graph node

        Args:
            Union['GraphNode', None]: new sequence of parent nodes
        """
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Returns graph node hash

        Returns:
            int: graph node hash
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns short node type description

        Returns:
            str: text graph node representation
        """
        pass

    def __repr__(self) -> str:
        """Returns full node description

        Returns:
            str: text graph node representation
        """
        return self.__str__()

    def description(self) -> str:
        """Returns full node description
        for use in recursive id.

        Returns:
            str: text graph node representation
        """

        return self.__str__()

    @property
    def descriptive_id(self) -> str:
        """Returns structural identifier of the subgraph starting at this node

        Returns:
            str: text description of the content in the node and its parameters
        """
        return descriptive_id_recursive(self)
