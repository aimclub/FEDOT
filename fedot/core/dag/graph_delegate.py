from copy import copy
from typing import Union, Sequence, List, Callable, Optional, Tuple

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.utilities.data_structures import Copyable


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
    def nodes(self) -> List[GraphNode]:
        return self.operator.nodes

    @nodes.setter
    def nodes(self, new_nodes: List[GraphNode]):
        self.operator.nodes = new_nodes

    @property
    def descriptive_id(self):
        return self.operator.descriptive_id

    @property
    def length(self) -> int:
        return self.operator.length

    @property
    def depth(self) -> int:
        return self.operator.depth
