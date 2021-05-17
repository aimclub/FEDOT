from copy import copy
from typing import Any, List, Optional

from fedot.core.graphs.node_operator import NodeOperator
from fedot.core.log import default_log


class GraphNode:
    """
    Base class for node definition in the DAG-based structure of GraphObject

    :param nodes_from: parent nodes which information comes from
    :param operation_type: str type of the operation in node
    :param log: Log object to record messages
    """

    def __init__(self, nodes_from: Optional[List['GraphNode']],
                 operation_type: Any,
                 log=None):
        self.nodes_from = nodes_from
        self.log = log
        self.operation = operation_type
        self._operator = NodeOperator(self)
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def __str__(self):
        operation = f'{self.operation}'
        return operation

    def __repr__(self):
        return self.__str__()

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        """
        Method returns verbal description of the operation in the node
        and its parameters
        """

        try:
            node_label = self.operation.description
        except AttributeError:
            node_label = self.operation

        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    def ordered_subnodes_hierarchy(self, visited=None) -> List['GraphNode']:
        return self._operator.ordered_subnodes_hierarchy(visited)

    @property
    def distance_to_primary_level(self):
        return self._operator.distance_to_primary_level()


class PrimaryGraphNode(GraphNode):
    """
    The class defines the interface of primary nodes located in the start of directed graph flow

    :param operation_type: str type of the operation in node
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: Any, **kwargs):
        super().__init__(nodes_from=None, operation_type=operation_type, **kwargs)


class SecondaryGraphNode(GraphNode):
    """
       The class defines the interface of intermediate and final nodes in directed graph

       :param operation_type: str type of the operation in node
       :param nodes_from: parent nodes
       :param kwargs: optional arguments (i.e. logger)
       """

    def __init__(self, operation_type: Any, nodes_from: Optional[List['GraphNode']] = None,
                 **kwargs):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, operation_type=operation_type,
                         **kwargs)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)
