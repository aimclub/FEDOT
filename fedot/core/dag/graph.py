from abc import ABC, abstractmethod
from os import PathLike
from typing import Tuple, Dict, List, Sequence, Union, TypeVar, Optional

from fedot.core.dag.graph_node import GraphNode

NodeType = TypeVar('NodeType', bound=GraphNode, covariant=False, contravariant=False)


class Graph(ABC):
    """Defines abstract graph interface that's required by graph optimisation process.
    """

    @abstractmethod
    def add_node(self, node: GraphNode):
        """Adds new node to the graph together with its parent nodes.

        Args:
            node: pipeline node
        """
        raise NotImplementedError()

    @abstractmethod
    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """Replaces ``old_node`` node with ``new_node``

        Args:
            old_node: node to be replaced
            new_node: node to be placed instead
        """
        raise NotImplementedError()

    @abstractmethod
    def update_subtree(self, old_subtree: GraphNode, new_subtree: GraphNode):
        """Changes ``old_subtree`` subtree to ``new_subtree``

        Args:
            old_subtree: node and its subtree to be removed
            new_subtree: node and its subtree to be placed instead
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_node(self, node: GraphNode):
        """Removes ``node`` from the graph.
        If ``node`` has only one child, then connects all of the ``node`` parents to it.

        Args:
            node: node of the graph to be deleted
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_subtree(self, subtree: GraphNode):
        """Deletes given node with all its parents.
        Deletes all edges from removed nodes to remaining graph nodes

        Args:
            subtree: node to be deleted with all of its parents
                and their connections amongst the remaining graph nodes
        """
        raise NotImplementedError()

    @abstractmethod
    def node_children(self, node: GraphNode) -> Sequence[Optional[GraphNode]]:
        """Returns all children of the ``node``

        Args:
            node: for getting children from

        Returns: children of the ``node``
        """
        raise NotImplementedError()

    @abstractmethod
    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        """Adds edge between ``parent`` and ``child``

        Args:
            node_parent: acts like parent in pipeline connection relations
            node_child:  acts like child in pipeline connection relations
        """
        raise NotImplementedError()

    @abstractmethod
    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         clean_up_leftovers: bool = True):
        """Removes an edge between two nodes

        Args:
            node_parent: where the removing edge comes out
            node_child: where the removing edge enters
            clean_up_leftovers: whether to remove the remaining invalid vertices with edges or not
        """
        raise NotImplementedError()

    @abstractmethod
    def get_edges(self) -> Sequence[Tuple[GraphNode, GraphNode]]:
        """Gets all available edges in this graph

        Returns:
            pairs of parent_node -> child_node
        """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other_graph: 'Graph') -> bool:
        """Compares this graph with the ``other_graph``

        Args:
            other_graph: another graph

        Returns:
            is it equal to ``other_graph`` in terms of the graphs
        """
        raise NotImplementedError()

    def root_nodes(self) -> Sequence[GraphNode]:
        raise NotImplementedError()

    @property
    def root_node(self) -> Union[GraphNode, Sequence[GraphNode]]:
        """Gets the final layer node(s) of the graph

        Returns:
            the final layer node(s)
        """
        roots = self.root_nodes()
        if len(roots) == 1:
            return roots[0]
        return roots

    @property
    @abstractmethod
    def nodes(self) -> List[GraphNode]:
        """Return list of all graph nodes

        Returns:
            graph nodes
        """
        raise NotImplementedError()

    @nodes.setter
    @abstractmethod
    def nodes(self, new_nodes: List[GraphNode]):
        raise NotImplementedError()

    @property
    @abstractmethod
    def depth(self) -> int:
        """Gets this graph depth from its sink-node to its source-node

        Returns:
            length of a path from the root node to the farthest primary node
        """
        raise NotImplementedError()

    @property
    def length(self) -> int:
        """Return size of the graph (number of nodes)

        Returns:
            graph size
        """

        return len(self.nodes)

    def show(self, save_path: Optional[Union[PathLike, str]] = None, **kwargs):
        """Visualizes graph or saves its picture to the specified ``path``

        Args:
            save_path: optional, save location of the graph visualization image.
        """

    @property
    def graph_description(self) -> Dict:
        """Return summary characteristics of the graph

        Returns:
            dict: containing information about the graph
        """
        return {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }

    @property
    def descriptive_id(self) -> str:
        """Returns human-readable identifier of the graph.

        Returns:
            str: text description of the content in the node and its parameters
        """
        return self.root_node.descriptive_id

    def __str__(self):
        return str(self.graph_description)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.length
