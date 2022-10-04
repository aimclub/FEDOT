from abc import ABC, abstractmethod
from typing import List, Optional, Union, Iterable, Hashable
from uuid import uuid4

from fedot.core.dag.graph_utils import node_depth, descriptive_id_recursive
from fedot.core.utilities.data_structures import UniqueList
from fedot.core.utils import DEFAULT_PARAMS_STUB


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
        return self._nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes: Optional[Iterable['DAGNode']]):
        self._nodes_from = UniqueList(nodes)

    def __hash__(self) -> int:
        return hash(self.uid)

    def __str__(self) -> str:
        return str(self.content['name'])

    def __repr__(self) -> str:
        return self.__str__()

    def description(self) -> str:
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
