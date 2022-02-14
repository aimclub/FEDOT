from typing import List, Optional, Union

from fedot.core.dag.node_operator import NodeOperator


class GraphNode:
    """
    Class for node definition in the DAG-based structure

    :param nodes_from: parent nodes which information comes from
    :param content: dict for the content in node
        The possible parameters are:
            'name' - name (str) or object that performs actions in this node
            'params' - dictionary with additional information that is used by
            the object in the 'name' field (e.g. hyperparameters values).
    """

    def __init__(self, content: Union[dict, str],
                 nodes_from: Optional[List['GraphNode']] = None):
        self.nodes_from = nodes_from
        # Wrap string into dict if it is necessary
        if isinstance(content, str):
            content = {'name': content}
        self.content = content
        self._operator = NodeOperator(self)

    def __str__(self):
        return str(self.content['name'])

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
