from copy import copy
from typing import TYPE_CHECKING, List, Optional, Union

from fedot.core.utils import DEFAULT_PARAMS_STUB

if TYPE_CHECKING:
    from fedot.core.dag.graph_node import GraphNode
    from fedot.core.optimisers.graph import OptNode

MAX_DEPTH = 1000


class NodeOperator:
    """
    _summary_

    :param node: bounded node to be processed by the operator
    """

    def __init__(self, node: Union['GraphNode', 'OptNode']):
        self._node = node

    def distance_to_primary_level(self) -> int:
        """
        Returns max depth from bounded node to graphs primary level

        :return: max depth to the primary level
        """
        if not self._node.nodes_from:
            return 0
        else:
            return 1 + max(next_node.distance_to_primary_level for next_node in self._node.nodes_from)

    def ordered_subnodes_hierarchy(self, visited: Optional[Union[List['GraphNode'], List['OptNode']]] = None) \
            -> Union[List['GraphNode'], List['OptNode']]:
        """
        Gets hierarchical subnodes representation of the graph starting from the bounded node

        :param visited: already visited nodes not to be included to the resulting hierarchical list

        :return: hierarchical subnodes list starting from the bounded node
        """
        if visited is None:
            visited = []

        if len(visited) > MAX_DEPTH:
            raise ValueError('Graph has cycle')
        nodes = [self._node]
        if self._node.nodes_from is not None:
            for parent in self._node.nodes_from:
                if parent not in visited:
                    visited.append(parent)
                    nodes.extend(parent.ordered_subnodes_hierarchy(visited))

        return nodes

    def descriptive_id(self) -> str:
        """
        Returns verbal identificator of the node

        :return: text description of the content in the node and its parameters
        """
        return _descriptive_id_recursive(self._node, visited_nodes=[])


def _descriptive_id_recursive(current_node: Union['GraphNode', 'OptNode'],
                              visited_nodes: Union[List['GraphNode'], List['OptNode']]) -> str:
    """
    Returns verbal identificator of the node

    :return: text description of the content in the node and its parameters
    """
    node_operation = current_node.content['name']
    params = current_node.content.get('params')
    if isinstance(node_operation, str):
        # If there is a string: name of operation (as in json repository)
        node_label = str(node_operation)
        if params and params != DEFAULT_PARAMS_STUB:
            node_label = f'n_{node_label}_{params}'
    else:
        # If instance of Operation is placed in 'name'
        node_label = node_operation.description(params)

    full_path = ''
    if current_node in visited_nodes:
        return 'ID_CYCLED'
    visited_nodes.append(current_node)
    if current_node.nodes_from:
        previous_items = []
        for parent_node in current_node.nodes_from:
            previous_items.append(f'{_descriptive_id_recursive(copy(parent_node), copy(visited_nodes))};')
        previous_items.sort()
        previous_items_str = ';'.join(previous_items)

        full_path += f'({previous_items_str})'
    full_path += f'/{node_label}'
    return full_path
