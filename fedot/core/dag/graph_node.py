from typing import Any, List, Optional

from fedot.core.dag.node_operator import NodeOperator


class GraphNode:
    """
    Class for node definition in the DAG-based structure

    :param nodes_from: parent nodes which information comes from
    :param content: str type of the content in node
    """

    def __init__(self, nodes_from: Optional[List['GraphNode']] = None,
                 content: Any = ''):
        self.nodes_from = nodes_from
        self.content = content
        self._operator = NodeOperator(self)

    def __str__(self):
        return str(self.content)

    def __repr__(self):
        return self.__str__()

    @property
    def descriptive_id(self):
        return self._operator.descriptive_id()

    def ordered_subnodes_hierarchy(self, visited=None) -> List['GraphNode']:
        return self._operator.ordered_subnodes_hierarchy(visited)

    @property
    def distance_to_primary_level(self):
        return self._operator.distance_to_primary_level()
