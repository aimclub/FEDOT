from typing import Iterable, List, Optional, Union
from uuid import uuid4

from fedot.core.dag.node_operator import NodeOperator
from fedot.core.utilities.data_structures import UniqueList
from fedot.core.utils import DEFAULT_PARAMS_STUB


class GraphNode:
    """Class for node definition in the DAG-based structure

    :param nodes_from: parent nodes which information comes from
    :param content: dict for the content in the node
        The possible parameters are:
            - 'name' - name (str) or object that performs actions in this node
            - 'params' - dictionary with additional information that is used by
                the object in the 'name' field (e.g. hyperparameters values)
    """

    def __init__(self, content: Union[dict, str],
                 nodes_from: Optional[List['GraphNode']] = None):
        # Wrap string into dict if it is necessary
        if isinstance(content, str):
            content = {'name': content}

        self.content = content
        self._nodes_from = UniqueList(nodes_from or ())
        self._operator = NodeOperator(self)
        self.uid = str(uuid4())

    def __str__(self):
        """Returns graph node description

        :return: text graph node representation
        """
        return str(self.content['name'])

    def __repr__(self):
        """Does the same as :meth:`__str__`

        :return: text graph node representation
        """
        return self.__str__()

    @property
    def nodes_from(self) -> List['GraphNode']:
        """Gets all parent nodes of this graph node

        :return: all the parent nodes
        :rtype: List[:class`~fedot.core.dag.graph_node.GraphNode`]
        """
        return self._nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes: Optional[Iterable['GraphNode']]):
        """Changes value of parent nodes of this graph node

        :param nodes: new sequence of parent nodes
        :type nodes: Iterable[:class`~fedot.core.dag.graph_node.GraphNode`] | None
        """
        self._nodes_from = UniqueList(nodes)

    @property
    def descriptive_id(self) -> str:
        """Returns verbal identificator of the node

        :return: text description of the content in the node and its parameters
        :rtype: str
        """
        return self._operator.descriptive_id()

    def ordered_subnodes_hierarchy(self, visited: Optional[List['GraphNode']] = None) -> List['GraphNode']:
        """Gets hierarchical subnodes representation of the graph starting from the bounded node

        :param visited: already visited nodes not to be included to the resulting hierarchical list

        :return: hierarchical subnodes list starting from the bounded node
        """
        return self._operator.ordered_subnodes_hierarchy(visited)

    @property
    def distance_to_primary_level(self) -> int:
        """
        Returns max depth from bounded node to graphs primary level

        :return: max depth to the primary level
        :rtype: int
        """
        return self._operator.distance_to_primary_level()
