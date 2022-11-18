from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional, Iterable
from uuid import uuid4


class GraphNode(ABC):
    """Definition of the node in directed graph structure.

    Provides interface for getting and modifying the parent nodes
    and recursive description based on all preceding nodes.
    """
    def __init__(self):
        self.uid = str(uuid4())

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
            nodes: new sequence of parent nodes
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """ Str name of this graph node """
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


def descriptive_id_recursive(current_node: GraphNode, visited_nodes=None) -> str:
    if visited_nodes is None:
        visited_nodes = []

    node_label = current_node.description()

    full_path_items = []
    if current_node in visited_nodes:
        return 'ID_CYCLED'
    visited_nodes.append(current_node)
    if current_node.nodes_from:
        previous_items = []
        for parent_node in current_node.nodes_from:
            previous_items.append(f'{descriptive_id_recursive(parent_node, copy(visited_nodes))};')
        previous_items.sort()
        previous_items_str = ';'.join(previous_items)

        full_path_items.append(f'({previous_items_str})')
    full_path_items.append(f'/{node_label}')
    full_path = ''.join(full_path_items)
    return full_path
