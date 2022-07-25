from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence, remove_items
from networkx import graph_edit_distance, set_node_attributes


class GraphOperator:
    """
    _summary_

    :param graph: object used as the :class:`~fedot.core.pipelines.pipeline.Pipeline` structure definition
        or as optimized structure
    :param nodes_postproc_func: nodes postprocessor after their modification
    """

    def __init__(self, graph: Optional[Union[Graph, OptGraph]] = None,
                 nodes_postproc_func: Optional[Callable[[List[GraphNode]], None]] = None):
        self._graph = graph
        self._postproc_nodes = nodes_postproc_func

    def delete_node(self, node: GraphNode):
        """
        Removes provided ``node`` from the bounded graph structure.
        If ``node`` has only one child connects all of the ``node`` parents to it

        :param node: node of the graph to be deleted
        """
        node_children_cached = self.node_children(node)
        self_root_node_cached = self._graph.root_node

        for node_child in self.node_children(node):
            node_child.nodes_from.remove(node)

        if node.nodes_from and len(node_children_cached) == 1:
            for node_from in node.nodes_from:
                node_children_cached[0].nodes_from.append(node_from)
        self._graph.nodes.clear()
        self.add_node(self_root_node_cached)
        self._postproc_nodes()

    def delete_subtree(self, node: GraphNode):
        """
        Deletes node with all the parents it has.
        Deletes all edges from removed nodes to remaining graph nodes

        :param node: node to be deleted with all of its parents and their connections amongst the remaining graph nodes
        """
        subtree_nodes = node.ordered_subnodes_hierarchy()
        self._graph._nodes = remove_items(self._graph.nodes, subtree_nodes)
        # prune all edges coming from the removed subtree
        for node in self._graph.nodes:
            node.nodes_from = remove_items(node.nodes_from, subtree_nodes)

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """
        Replaces ``old_node`` with ``new_node``

        :param old_node: node to be replaced
        :param new_node: node to be placed instead
        """
        self.actualise_old_node_children(old_node, new_node)
        if old_node.nodes_from:
            if new_node.nodes_from:
                # extend sources of new_node with sources of old node
                new_node.nodes_from.extend(old_node.nodes_from)
            else:
                # just assign old sources as sources for the new node
                new_node.nodes_from = old_node.nodes_from
        self._graph.nodes.remove(old_node)
        self._graph.nodes.append(new_node)
        self.sort_nodes()
        self._postproc_nodes()

    def update_subtree(self, old_node: GraphNode, new_node: GraphNode):
        """Exchange subtrees with old and new nodes as roots of subtrees"""
        new_node = deepcopy(new_node)
        self.actualise_old_node_children(old_node, new_node)
        self.delete_subtree(old_node)
        self.add_node(new_node)
        self.sort_nodes()

    def add_node(self, node: GraphNode):
        """
        Add new node to the Pipeline

        :param node: new Node object
        """
        if node not in self._graph.nodes:
            self._graph.nodes.append(node)
            if node.nodes_from:
                for new_parent_node in node.nodes_from:
                    self.add_node(new_parent_node)

    def distance_to_root_level(self, node: GraphNode):
        def recursive_child_height(parent_node: GraphNode) -> int:
            node_child = self.node_children(parent_node)
            if node_child:
                height = recursive_child_height(node_child[0]) + 1
                return height
            else:
                return 0

        height = recursive_child_height(node)
        return height

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        def get_nodes(node: Any, current_height):
            nodes = []
            if current_height == layer_number:
                nodes.append(node)
            else:
                if node.nodes_from:
                    for child in node.nodes_from:
                        nodes.extend(get_nodes(child, current_height + 1))
            return nodes

        nodes = get_nodes(self._graph.root_node, current_height=0)
        return nodes

    def actualise_old_node_children(self, old_node: GraphNode, new_node: GraphNode):
        old_node_offspring = self.node_children(old_node)
        for old_node_child in old_node_offspring:
            updated_index = old_node_child.nodes_from.index(old_node)
            old_node_child.nodes_from[updated_index] = new_node

    def sort_nodes(self):
        """ Layer by layer sorting """
        if isinstance(self._graph.root_node, list):
            nodes = self._graph.nodes
        else:
            nodes = self._graph.root_node.ordered_subnodes_hierarchy()
        self._graph._nodes = nodes

    def node_children(self, node: GraphNode) -> List[Optional[GraphNode]]:
        return [other_node for other_node in self._graph.nodes
                if other_node.nodes_from and
                node in other_node.nodes_from]

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
        """
        Method removes nodes and edges that do not affect the result of the pipeline

        Leftovers - edges and nodes that remain after the removal of the edge / node
        and do not affect the result of the pipeline
        """

        if not self.node_children(node):
            self._graph.nodes.remove(node)
            if node.nodes_from:
                for node in node.nodes_from:
                    self._clean_up_leftovers(node)

    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         is_clean_up_leftovers: bool = True):
        """
        Method to remove an edge between two nodes

        :param node_parent: the node from which the removing edge comes out
        :param node_child: the node in which the removing edge enters
        :param is_clean_up_leftovers: bool flag whether to remove the remaining
        invalid vertices and edges or not
        """

        if node_child.nodes_from is None or node_parent not in node_child.nodes_from:
            return
        elif node_parent not in self._graph.nodes or node_child not in self._graph.nodes:
            return
        elif len(node_child.nodes_from) == 1:
            node_child.nodes_from = None
        else:
            node_child.nodes_from.remove(node_parent)

        if is_clean_up_leftovers:
            self._clean_up_leftovers(node_parent)

        self._postproc_nodes()

    def root_node(self) -> Union[GraphNode, List[GraphNode]]:
        if not self._graph.nodes:
            return []
        roots = [node for node in self._graph.nodes
                 if not any(self._graph.node_children(node))]
        if len(roots) == 1:
            return roots[0]
        return roots

    def is_graph_equal(self, other_graph: Union['Graph', 'OptGraph']) -> bool:
        """
        Compares this graph with the ``other_graph``

        :param other_graph: another graph

        :return: is it equal to ``other_graph`` in terms of the graphs
        """
        if all(isinstance(rn, list) for rn in [self._graph.root_node, other_graph.root_node]):
            return set(rn.descriptive_id for rn in self._graph.root_node) == \
                   set(rn.descriptive_id for rn in other_graph.root_node)
        elif all(not isinstance(rn, list) for rn in [self._graph.root_node, other_graph.root_node]):
            return self._graph.root_node.descriptive_id == other_graph.root_node.descriptive_id
        else:
            return False

    def graph_description(self) -> str:
        """
        Returns graph description

        :return: text graph representation
        """
        return str({
            'depth': self._graph.depth,
            'length': self._graph.length,
            'nodes': self._graph.nodes,
        })

    @property
    def descriptive_id(self) -> str:
        root_list = ensure_wrapped_in_sequence(self.root_node())
        full_desc_id = ''.join([r.descriptive_id for r in root_list])
        return full_desc_id

    def graph_depth(self) -> int:
        if not self._graph.nodes:
            return 0

        def _depth_recursive(node: GraphNode):
            if node is None:
                return 0
            if node.nodes_from is None or not node.nodes_from:
                return 1
            else:
                return 1 + max(_depth_recursive(next_node) for next_node in node.nodes_from)

        root = ensure_wrapped_in_sequence(self.root_node())
        return max(_depth_recursive(n) for n in root)

    def get_nodes_degrees(self):
        """ Nodes degree as the number of edges the node has:
         k = k(in) + k(out) """
        graph, _ = graph_structure_as_nx_graph(self._graph)
        index_degree_pairs = graph.degree
        node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
        return node_degrees

    def get_edges(self) -> List[Tuple[GraphNode, GraphNode]]:
        """ Returns all available edges in a given graph """
        edges = []
        for node in self._graph.nodes:
            if node.nodes_from:
                for parent_node in node.nodes_from:
                    edges.append((parent_node, node))
        return edges


def get_distance_between(graph_1: 'Graph', graph_2: 'Graph') -> int:
    """ Returns distance between specified graphs """

    def node_match(node_data_1: Dict[str, GraphNode], node_data_2: Dict[str, GraphNode]) -> bool:
        node_1, node_2 = node_data_1.get('node'), node_data_2.get('node')

        is_operation_match = str(node_1) == str(node_2)
        is_params_match = node_1.content.get('params') == node_2.content.get('params')
        is_match = is_operation_match and is_params_match
        return is_match

    graphs = (graph_1, graph_2)
    nx_graphs = []
    for graph in graphs:
        nx_graph, nodes = graph_structure_as_nx_graph(graph)
        set_node_attributes(nx_graph, nodes, name='node')
        nx_graphs.append(nx_graph)

    distance = graph_edit_distance(*nx_graphs, node_match=node_match)
    return int(distance)
