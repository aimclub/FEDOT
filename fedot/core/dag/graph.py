from os import PathLike
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Sequence, Union, TypeVar, Generic, Optional

from fedot.core.dag.graph_node import GraphNode
from fedot.core.visualisation.graph_viz import GraphVisualiser, NodeColorType

NodeType = TypeVar('NodeType', bound=GraphNode, covariant=False, contravariant=False)


class Graph(ABC):
    """
    Defines abstract graph interface that's required by graph optimisation process.
    """

    @abstractmethod
    def add_node(self, new_node: GraphNode):
        """
        Add new node to the Pipeline

        :param new_node: new GraphNode object
        """
        raise NotImplementedError()

    @abstractmethod
    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """
        Replace old_node with new one.

        :param old_node: 'GraphNode' object to replace
        :param new_node: 'GraphNode' new object
        """
        raise NotImplementedError()

    @abstractmethod
    def update_subtree(self, old_subroot: GraphNode, new_subroot: GraphNode):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: 'GraphNode' object to replace
        :param new_subroot: 'GraphNode' new object
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_node(self, node: GraphNode):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: 'GraphNode' object to delete
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_subtree(self, subroot: GraphNode):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        raise NotImplementedError()

    @abstractmethod
    def distance_to_root_level(self, node: GraphNode) -> int:
        """ Returns distance to root level """
        raise NotImplementedError()

    @abstractmethod
    def nodes_from_layer(self, layer_number: int) -> Sequence[GraphNode]:
        """ Returns all nodes from specified layer """
        raise NotImplementedError()

    @abstractmethod
    def node_children(self, node: GraphNode) -> Sequence[Optional[GraphNode]]:
        """ Returns all node's children """
        raise NotImplementedError()

    @abstractmethod
    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        """ Add an edge from node_parent to node_child """
        raise NotImplementedError()

    @abstractmethod
    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         is_clean_up_leftovers: bool = True):
        """ Delete an edge from node_parent to node_child """
        raise NotImplementedError()

    @abstractmethod
    def get_nodes_degrees(self):
        """ Nodes degree as the number of edges the node has: k = k(in) + k(out) """
        raise NotImplementedError()

    @abstractmethod
    def get_edges(self) -> List[Tuple[GraphNode, GraphNode]]:
        """ Returns all available edges in a given graph """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()

    def __str__(self):
        return str(self.graph_description)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.length

    @property
    def root_node(self) -> Union[GraphNode, Sequence[GraphNode]]:
        raise NotImplementedError()

    @property
    def nodes(self) -> Sequence[GraphNode]:
        raise NotImplementedError()

    @property
    def descriptive_id(self):
        raise NotImplementedError()

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        raise NotImplementedError()

    def show(self, save_path: Optional[Union[PathLike, str]] = None, engine: str = 'matplotlib',
             node_color: Optional[NodeColorType] = None, dpi: int = 300,
             node_size_scale: float = 1.0, font_size_scale: float = 1.0, edge_curvature_scale: float = 1.0):
        """Visualizes graph or saves its picture to the specified ``path``

        :param save_path: optional, save location of the graph visualization image.
        :param engine: engine to visualize the graph. Possible values: 'matplotlib', 'pyvis', 'graphviz'.
        :param node_color: color of nodes to use.
        :param node_size_scale: use to make node size bigger or lesser. Supported only for the engine 'matplotlib'.
        :param font_size_scale: use to make font size bigger or lesser. Supported only for the engine 'matplotlib'.
        :param edge_curvature_scale: use to make edges more or less curved. Supported only for the engine 'matplotlib'.
        :param dpi: DPI of the output image. Not supported for the engine 'pyvis'.
        """
        GraphVisualiser().visualise(self, save_path, engine, node_color, dpi, node_size_scale, font_size_scale,
                                    edge_curvature_scale)

    @property
    def graph_description(self) -> Dict:
        return {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }


class GraphDelegate(Graph):
    """
    Graph that delegates calls to another Graph implementation.

    The class purpose is for cleaner code organisation:
    - avoid inheriting from specific Graph implementations
    - hide Graph implementation details from inheritors.

    :param delegate: Graph implementation to delegate to.
    """

    def __init__(self, delegate: Graph):
        self.operator = delegate

    def add_node(self, new_node: GraphNode):
        self.operator.add_node(new_node)

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        self.operator.update_node(old_node, new_node)

    def update_subtree(self, old_subroot: GraphNode, new_subroot: GraphNode):
        self.operator.update_subtree(old_subroot, new_subroot)

    def delete_node(self, node: GraphNode):
        self.operator.delete_node(node)

    def delete_subtree(self, subroot: GraphNode):
        self.operator.delete_subtree(subroot)

    def distance_to_root_level(self, node: GraphNode) -> int:
        return self.operator.distance_to_root_level(node=node)

    def nodes_from_layer(self, layer_number: int) -> Sequence[GraphNode]:
        return self.operator.nodes_from_layer(layer_number=layer_number)

    def node_children(self, node: GraphNode) -> Sequence[Optional[GraphNode]]:
        return self.operator.node_children(node=node)

    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        self.operator.connect_nodes(node_parent, node_child)

    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         is_clean_up_leftovers: bool = True):
        self.operator.disconnect_nodes(node_parent, node_child, is_clean_up_leftovers)

    def get_nodes_degrees(self):
        return self.operator.get_nodes_degrees()

    def get_edges(self) -> Sequence[Tuple[GraphNode, GraphNode]]:
        return self.operator.get_edges()

    def __eq__(self, other) -> bool:
        return self.operator.__eq__(other)

    def __str__(self):
        return self.operator.__str__()

    def __repr__(self):
        return self.operator.__repr__()

    @property
    def root_node(self) -> Union[GraphNode, Sequence[GraphNode]]:
        return self.operator.root_node

    @property
    def nodes(self) -> Sequence[GraphNode]:
        return self.operator.nodes

    @property
    def descriptive_id(self):
        return self.operator.descriptive_id

    @property
    def length(self) -> int:
        return self.operator.length

    @property
    def depth(self) -> int:
        return self.operator.depth
