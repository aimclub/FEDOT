from os import PathLike
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, List, Sequence, Union, TypeVar, Generic, Optional

from fedot.core.dag.graph_node import GraphNode
from fedot.core.visualisation.graph_viz import GraphVisualiser, NodeColorType

NodeType = TypeVar('NodeType', bound=GraphNode, covariant=False, contravariant=False)


class Graph(ABC):
    """Defines abstract graph interface that's required by graph optimisation process.
    """

    @abstractmethod
    def add_node(self, node: GraphNode):
        """Adds new node to the graph together with its parent nodes.

        Args:
            nodes: pipeline nodes
        """

    @abstractmethod
    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """Replaces ``old_node`` node with ``new_node``

        :param old_node: node to be replaced
        :param new_node: node to be placed instead
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
    def delete_subtree(self, subroot: GraphNode):
        """Deletes given node with all its parents.
        Deletes all edges from removed nodes to remaining graph nodes

        Args:
            subtree: node to be deleted with all of its parents
                and their connections amongst the remaining graph nodes
        """
        raise NotImplementedError()

    @abstractmethod
    def distance_to_root_level(self, node: GraphNode) -> int:
        """Gets distance to the final output node

        Args:
            node: search starting point
        """
        raise NotImplementedError()

    @abstractmethod
    def nodes_from_layer(self, layer_number: int) -> Sequence[GraphNode]:
        """Gets all the nodes from the chosen layer up to the surface

        Args:
            layer_number: max height of diving

        Returns:
            all nodes from the surface to the ``layer_number`` layer
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
    def get_nodes_degrees(self) -> Sequence[int]:
        """Nodes degree as the number of edges the node has:
            ``degree = #input_edges + #out_edges``

        Returns:
            nodes degrees ordered according to the nx_graph representation of this graph
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

    @property
    @abstractmethod
    def root_node(self) -> Union[GraphNode, Sequence[GraphNode]]:
        """Gets the final layer node(s) of the graph

        Returns:
            the final layer node(s)
        """
        raise NotImplementedError()

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

    def show(self, save_path: Optional[Union[PathLike, str]] = None, engine: str = 'matplotlib',
             node_color: Optional[NodeColorType] = None, dpi: int = 300,
             node_size_scale: float = 1.0, font_size_scale: float = 1.0, edge_curvature_scale: float = 1.0):
        """Visualizes graph or saves its picture to the specified ``path``

        Args:
            save_path: optional, save location of the graph visualization image.
            engine: engine to visualize the graph. Possible values: 'matplotlib', 'pyvis', 'graphviz'.
            node_color: color of nodes to use.
            node_size_scale: use to make node size bigger or lesser. Supported only for the engine 'matplotlib'.
            font_size_scale: use to make font size bigger or lesser. Supported only for the engine 'matplotlib'.
            edge_curvature_scale: use to make edges more or less curved. Supported only for the engine 'matplotlib'.
            dpi: DPI of the output image. Not supported for the engine 'pyvis'.
        """
        GraphVisualiser().visualise(self, save_path, engine, node_color, dpi, node_size_scale, font_size_scale,
                                    edge_curvature_scale)

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
