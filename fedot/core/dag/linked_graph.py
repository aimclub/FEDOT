from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence

from networkx import graph_edit_distance, set_node_attributes

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_utils import ordered_subnodes_hierarchy, node_depth
from fedot.core.dag.convert import graph_structure_as_nx_graph
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence, Copyable, remove_items
from fedot.core.utils import copy_doc

NodePostprocessCallable = Callable[[Graph, Sequence[GraphNode]], Any]


class LinkedGraph(Graph, Copyable):
    """Graph implementation based on linked graph node
    that directly stores its parent nodes.

    Args:
        nodes: nodes of the Graph
        postprocess_nodes: nodes postprocessing function used after their modification
    """

    def __init__(self, nodes: Union[GraphNode, Sequence[GraphNode]] = (),
                 postprocess_nodes: Optional[NodePostprocessCallable] = None):
        self._nodes = []
        for node in ensure_wrapped_in_sequence(nodes):
            self.add_node(node)
        self._postprocess_nodes = postprocess_nodes or self._empty_postprocess

    @staticmethod
    def _empty_postprocess(*args):
        pass

    @copy_doc(Graph)
    def delete_node(self, node: GraphNode):
        node_children_cached = self.node_children(node)
        self_root_node_cached = self.root_node

        for node_child in self.node_children(node):
            node_child.nodes_from.remove(node)

        if node.nodes_from and len(node_children_cached) == 1:
            for node_from in node.nodes_from:
                node_children_cached[0].nodes_from.append(node_from)
        self._nodes.clear()
        self.add_node(self_root_node_cached)
        self._postprocess_nodes(self, self._nodes)

    @copy_doc(Graph)
    def delete_subtree(self, subtree: GraphNode):
        subtree_nodes = ordered_subnodes_hierarchy(subtree)
        self._nodes = remove_items(self._nodes, subtree_nodes)
        # prune all edges coming from the removed subtree
        for subtree in self._nodes:
            subtree.nodes_from = remove_items(subtree.nodes_from, subtree_nodes)

    @copy_doc(Graph)
    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        self.actualise_old_node_children(old_node, new_node)
        new_node.nodes_from.extend(old_node.nodes_from)
        self._nodes.remove(old_node)
        self._nodes.append(new_node)
        self.sort_nodes()
        self._postprocess_nodes(self, self._nodes)

    @copy_doc(Graph)
    def update_subtree(self, old_subtree: GraphNode, new_subtree: GraphNode):
        new_subtree = deepcopy(new_subtree)
        self.actualise_old_node_children(old_subtree, new_subtree)
        self.delete_subtree(old_subtree)
        self.add_node(new_subtree)
        self.sort_nodes()

    @copy_doc(Graph)
    def add_node(self, node: GraphNode):
        if node not in self._nodes:
            self._nodes.append(node)
            for n in node.nodes_from:
                self.add_node(n)

    def actualise_old_node_children(self, old_node: GraphNode, new_node: GraphNode):
        """Changes parent of ``old_node`` children to ``new_node``

        :param old_node: node to take children from
        :param new_node: new parent of ``old_node`` children
        """
        old_node_offspring = self.node_children(old_node)
        for old_node_child in old_node_offspring:
            updated_index = old_node_child.nodes_from.index(old_node)
            old_node_child.nodes_from[updated_index] = new_node

    def sort_nodes(self):
        """ Layer by layer sorting """
        if not isinstance(self.root_node, Sequence):
            self._nodes = ordered_subnodes_hierarchy(self.root_node)

    @copy_doc(Graph)
    def node_children(self, node: GraphNode) -> List[Optional[GraphNode]]:
        return [other_node for other_node in self._nodes
                if other_node.nodes_from and
                node in other_node.nodes_from]

    @copy_doc(Graph)
    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        if node_child in self.node_children(node_parent):
            return
        node_child.nodes_from.append(node_parent)

    def _clean_up_leftovers(self, node: GraphNode):
        """Removes nodes and edges that do not affect the result of the pipeline.
        Leftovers are edges and nodes that remain after the removal of the edge / node
            and do not affect the result of the pipeline.

        :param node: node to be deleted with all of its parents
        """

        if not self.node_children(node):
            self._nodes.remove(node)
            for node in node.nodes_from:
                self._clean_up_leftovers(node)

    @copy_doc(Graph)
    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         clean_up_leftovers: bool = True):
        if node_parent not in node_child.nodes_from:
            return
        if node_parent not in self._nodes or node_child not in self._nodes:
            return
        node_child.nodes_from.remove(node_parent)
        if clean_up_leftovers:
            self._clean_up_leftovers(node_parent)
        self._postprocess_nodes(self, self._nodes)

    def root_nodes(self) -> Sequence[GraphNode]:
        return [node for node in self._nodes if not any(self.node_children(node))]

    @property
    def nodes(self) -> List[GraphNode]:
        return self._nodes

    @nodes.setter
    def nodes(self, new_nodes: List[GraphNode]):
        self._nodes = new_nodes

    @copy_doc(Graph)
    def __eq__(self, other_graph: Graph) -> bool:
        return \
            set(rn.descriptive_id for rn in self.root_nodes()) == \
            set(rn.descriptive_id for rn in other_graph.root_nodes())

    @copy_doc(Graph)
    @property
    def descriptive_id(self) -> str:
        return ''.join([r.descriptive_id for r in self.root_nodes()])

    @copy_doc(Graph)
    @property
    def depth(self) -> int:
        return 0 if not self._nodes else max(map(node_depth, self.root_nodes()))

    @copy_doc(Graph)
    def get_edges(self) -> Sequence[Tuple[GraphNode, GraphNode]]:
        edges = []
        for node in self._nodes:
            if node.nodes_from:
                for parent_node in node.nodes_from:
                    edges.append((parent_node, node))
        return edges


def get_distance_between(graph_1: Graph, graph_2: Graph) -> int:
    """
    Gets edit distance from ``graph_1`` graph to the ``graph_2``

    :param graph_1: left object to compare
    :param graph_2: right object to compare

    :return: graph edit distance (aka Levenstein distance for graphs)
    """

    def node_match(node_data_1: Dict[str, GraphNode], node_data_2: Dict[str, GraphNode]) -> bool:
        """Checks if the two given nodes are identical

        :param node_data_1: nx_graph format for the first node to compare
        :param node_data_2: nx_graph format for the second node to compare

        :return: is the first node equal to the second
        """
        node_1, node_2 = node_data_1.get('node'), node_data_2.get('node')

        operations_do_match = str(node_1) == str(node_2)
        params_do_match = node_1.content.get('params') == node_2.content.get('params')
        nodes_do_match = operations_do_match and params_do_match
        return nodes_do_match

    graphs = (graph_1, graph_2)
    nx_graphs = []
    for graph in graphs:
        nx_graph, nodes = graph_structure_as_nx_graph(graph)
        set_node_attributes(nx_graph, nodes, name='node')
        nx_graphs.append(nx_graph)

    distance = graph_edit_distance(*nx_graphs, node_match=node_match)
    return int(distance)
