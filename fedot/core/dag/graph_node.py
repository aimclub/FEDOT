from copy import copy
from typing import List, Optional, Union, Iterable
from uuid import uuid4

from fedot.core.utilities.data_structures import UniqueList


class GraphNode:
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
                 nodes_from: Optional[List['GraphNode']] = None):
        # Wrap string into dict if it is necessary
        if isinstance(content, str):
            content = {'name': content}

        self.content = content
        self._nodes_from = UniqueList(nodes_from or ())
        self.uid = str(uuid4())

    def __str__(self):
        """Returns graph node description

        Returns:
            text graph node representation
        """

        return str(self.content['name'])

    def __repr__(self):
        """Does the same as :meth:`__str__`

        Returns:
            text graph node representation
        """

        return self.__str__()

    @property
    def nodes_from(self) -> List['GraphNode']:
        """Gets all parent nodes of this graph node

        Returns:
            List['GraphNode']: all the parent nodes
        """

        return self._nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes: Optional[Iterable['GraphNode']]):
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


def _descriptive_id_recursive(current_node: GraphNode, visited_nodes: Optional[List[GraphNode]] = None) -> str:
    """Method returns verbal description of the content in the node
    and its parameters
    """

    if visited_nodes is None:
        visited_nodes = []

    node_operation = current_node.content['name']
    params = current_node.content.get('params')
    if isinstance(node_operation, str):
        # If there is a string: name of operation (as in json repository)
        node_label = str(node_operation)
        if params:
            node_label = f'n_{node_label}_{params}'
    else:
        # If instance of Operation is placed in 'name'
        node_label = node_operation.description(params)

    full_path_items = []
    if current_node in visited_nodes:
        return 'ID_CYCLED'
    visited_nodes.append(current_node)
    if current_node.nodes_from:
        previous_items = []
        for parent_node in current_node.nodes_from:
            previous_items.append(f'{_descriptive_id_recursive(copy(parent_node), copy(visited_nodes))};')
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
