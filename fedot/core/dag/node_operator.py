from copy import copy
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fedot.core.dag.graph_node import GraphNode

MAX_DEPTH = 100


class NodeOperator:
    def __init__(self, node):
        self._node = node

    def distance_to_primary_level(self):
        if not self._node.nodes_from:
            return 0
        else:
            return 1 + max([next_node.distance_to_primary_level for next_node in self._node.nodes_from])

    def ordered_subnodes_hierarchy(self, visited=None) -> List['GraphNode']:
        if visited is None:
            visited = []

        if len(visited) > MAX_DEPTH:
            raise ValueError('Graph has cycle')
        nodes = [self._node]
        if self._node.nodes_from:
            for parent in self._node.nodes_from:
                if parent not in visited:
                    visited.append(parent)
                    nodes.extend(parent.ordered_subnodes_hierarchy(visited))

        return nodes

    def descriptive_id(self) -> str:
        return _descriptive_id_recursive(self._node, visited_nodes=[])


def _descriptive_id_recursive(current_node, visited_nodes) -> str:
    """
    Method returns verbal description of the content in the node
    and its parameters
    """
    if isinstance(current_node.content['name'], str):
        # If there is a string: name of operation (as in json repository)
        node_label = current_node.content['name']
        if current_node.content.get('params'):
            node_label = f'n_{node_label}_{current_node.content["params"]}'
    else:
        # If instance of Operation is placed in 'name'
        operation_params = current_node.content.get('params')
        node_label = current_node.content['name'].description(operation_params)

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
