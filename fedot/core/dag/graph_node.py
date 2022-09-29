from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional, Union, Iterable
from uuid import uuid4

from fedot.core.utilities.data_structures import UniqueList
from fedot.core.utils import DEFAULT_PARAMS_STUB


class GraphNode(ABC):
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

        Returns:
            Union['GraphNode', None]: new sequence of parent nodes
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
        return _descriptive_id_recursive(self)

    @property
    def distance_to_primary_level(self) -> int:
        """Returns max depth from bounded node to graphs primary level

        Returns:
            int: max depth to the primary level
        """
        return node_depth(self) - 1


class DAGNode(GraphNode):
    """Class for node definition in the DAG-based structure

    Args:
        nodes_from: parent nodes which information comes from
        content: ``dict`` for the content in the node

    Notes:
        The possible parameters are:
            - ``name`` - name (str) or object that performs actions in this node
            - ``params`` - dictionary with additional information that is used by
                    the object in the ``name`` field (e.g. hyperparameters values)
    """

    def __init__(self, content: Union[dict, str],
                 nodes_from: Optional[Iterable['DAGNode']] = None):
        # Wrap string into dict if it is necessary
        if isinstance(content, str):
            content = {'name': content}

        self.content = content
        self._nodes_from = UniqueList(nodes_from or ())
        self.uid = str(uuid4())

    @property
    def nodes_from(self) -> List['DAGNode']:
        """Gets all parent nodes of this graph node

        Returns:
            List['GraphNode']: all the parent nodes
        """

        return self._nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes: Optional[Iterable['DAGNode']]):
        """Changes value of parent nodes of this graph node

        Returns:
            Union['GraphNode', None]: new sequence of parent nodes
        """

        self._nodes_from = UniqueList(nodes)

    @property
    def descriptive_id(self) -> str:
        """Returns verbal identificator of the node

        Returns:
            str: text description of the content in the node and its parameters
        """
        return _descriptive_id_recursive(self)

    @property
    def distance_to_primary_level(self) -> int:
        """Returns max depth from bounded node to graphs primary level

        Returns:
            int: max depth to the primary level
        """
        return node_depth(self) - 1

    def __str__(self) -> str:
        """Returns short node type description

        Returns:
            str: text graph node representation
        """

        return str(self.content['name'])

    def __repr__(self) -> str:
        return self.__str__()

    def description(self) -> str:
        """Returns full node description

        Returns:
            str: text graph node representation
        """

        node_operation = self.content['name']
        params = self.content.get('params')
        # TODO: possibly unify with __repr__ & don't duplicate Operation.description
        if isinstance(node_operation, str):
            # If there is a string: name of operation (as in json repository)
            if params and params != DEFAULT_PARAMS_STUB:
                node_label = f'n_{node_operation}_{params}'
            else:
                node_label = f'n_{node_operation}'
        else:
            # If instance of Operation is placed in 'name'
            node_label = node_operation.description(params)
        return node_label


def _descriptive_id_recursive(current_node, visited_nodes=None) -> str:
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
            previous_items.append(f'{_descriptive_id_recursive(parent_node, copy(visited_nodes))};')
        previous_items.sort()
        previous_items_str = ';'.join(previous_items)

        full_path_items.append(f'({previous_items_str})')
    full_path_items.append(f'/{node_label}')
    full_path = ''.join(full_path_items)
    return full_path


def ordered_subnodes_hierarchy(node: 'GraphNode') -> List['GraphNode']:
    """Gets hierarchical subnodes representation of the graph starting from the bounded node

    Returns:
        List['GraphNode']: hierarchical subnodes list starting from the bounded node
    """
    started = {node}
    visited = set()

    def subtree_impl(node):
        nodes = [node]
        for parent in node.nodes_from:
            if parent in visited:
                continue
            elif parent in started:
                raise ValueError('Can not build ordered node hierarchy: graph has cycle')
            started.add(parent)
            nodes.extend(subtree_impl(parent))
            visited.add(parent)
        return nodes

    return subtree_impl(node)


def node_depth(node: GraphNode) -> int:
    """Gets this graph depth from the provided ``node`` to the graph source node

    Args:
        node: where to start diving from

    Returns:
        int: length of a path from the provided ``node`` to the farthest primary node
    """
    if not node.nodes_from:
        return 1
    else:
        return 1 + max(node_depth(next_node) for next_node in node.nodes_from)
