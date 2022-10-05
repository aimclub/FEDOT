from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence

from networkx import graph_edit_distance, set_node_attributes

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence, remove_items, Copyable
from fedot.core.utils import copy_doc

NodePostprocessCallable = Callable[[Graph, Sequence[GraphNode]], Any]


class GraphOperator(Graph, Copyable):
    """_summary_

    :param nodes: nodes of the Graph
    :param postprocess_nodes: nodes postprocessor after their modification
    """

    def __init__(self, nodes: Union[GraphNode, Sequence[GraphNode]] = (),
                 postprocess_nodes: Optional[NodePostprocessCallable] = None):
        self._nodes = []
        for node in ensure_wrapped_in_sequence(nodes):
            self.add_node(node)
        self._postprocess_nodes = postprocess_nodes or self._empty_postprocess

    def _empty_postprocess(self, *args):
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
        subtree_nodes = subtree.ordered_subnodes_hierarchy()
        self._nodes = remove_items(self._nodes, subtree_nodes)
        # prune all edges coming from the removed subtree
        for subtree in self._nodes:
            subtree.nodes_from = remove_items(subtree.nodes_from, subtree_nodes)

    @copy_doc(Graph)
    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        self.actualise_old_node_children(old_node, new_node)
        if old_node.nodes_from:
            if new_node.nodes_from:
                # extend sources of new_node with sources of old node
                new_node.nodes_from.extend(old_node.nodes_from)
            else:
                # just assign old sources as sources for the new node
                new_node.nodes_from = old_node.nodes_from
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
            if node.nodes_from:
                for new_parent_node in node.nodes_from:
                    self.add_node(new_parent_node)

    @copy_doc(Graph)
    def distance_to_root_level(self, node: GraphNode) -> int:

        def recursive_child_height(parent_node: GraphNode) -> int:
            """Recursively dives into ``parent_node`` children to get the bottom height

            :param node: search starting point
            """
            node_child = self.node_children(parent_node)
            if node_child:
                height = recursive_child_height(node_child[0]) + 1
                return height
            return 0

        height = recursive_child_height(node)
        return height

    @copy_doc(Graph)
    def nodes_from_layer(self, layer_number: int) -> List[Any]:

        def get_nodes(node: Union[GraphNode, List[GraphNode]], current_height: int):
            """Gets all the parent nodes of ``node``

            :param node: node to get all subnodes from
            :param current_height: current diving step depth

            :return: all parent nodes of ``node``
            """
            nodes = []
            if current_height == layer_number:
                nodes.append(node)
            else:
                if node.nodes_from:
                    for child in node.nodes_from:
                        nodes.extend(get_nodes(child, current_height + 1))
            return nodes

        nodes = get_nodes(self.root_node, current_height=0)
        return nodes

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
            self._nodes = self.root_node.ordered_subnodes_hierarchy()

    @copy_doc(Graph)
    def node_children(self, node: GraphNode) -> List[Optional[GraphNode]]:
        return [other_node for other_node in self._nodes
                if other_node.nodes_from and
                node in other_node.nodes_from]

    @copy_doc(Graph)
    def connect_nodes(self, parent: GraphNode, child: GraphNode):
        if child in self.node_children(parent):
            return
        # if not already connected
        if child.nodes_from:
            child.nodes_from.append(parent)
        else:
            # add parent to initial node
            new_child = GraphNode(nodes_from=[], content=child.content)
            new_child.nodes_from.append(parent)
            self.update_node(child, new_child)

    def _clean_up_leftovers(self, node: GraphNode):
        """Removes nodes and edges that do not affect the result of the pipeline.
        Leftovers are edges and nodes that remain after the removal of the edge / node
            and do not affect the result of the pipeline.

        :param node: node to be deleted with all of its parents
        """

        if not self.node_children(node):
            self._nodes.remove(node)
            if node.nodes_from:
                for node in node.nodes_from:
                    self._clean_up_leftovers(node)

    @copy_doc(Graph)
    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         clean_up_leftovers: bool = True):
        if not node_child.nodes_from or node_parent not in node_child.nodes_from:
            return
        elif node_parent not in self._nodes or node_child not in self._nodes:
            return
        elif len(node_child.nodes_from) == 1:
            node_child.nodes_from = None
        else:
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
        if not self._nodes:
            return 0

        def _depth_recursive(node: GraphNode) -> int:
            """Gets this graph depth from the provided ``node`` to the graph source node

            :param node: where to start diving from

            :return: length of a path from the provided ``node`` to the farthest primary node
            """
            if node is None:  # is it real situation to have None in `node.nodes_from`?
                return 0
            if not node.nodes_from:
                return 1
            else:
                return 1 + max(_depth_recursive(next_node) for next_node in node.nodes_from)

        root = ensure_wrapped_in_sequence(self.root_node)
        return max(_depth_recursive(n) for n in root)

    @copy_doc(Graph)
    def get_nodes_degrees(self) -> Sequence[int]:
        graph, _ = graph_structure_as_nx_graph(self)
        index_degree_pairs = graph.degree
        node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
        return node_degrees

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
