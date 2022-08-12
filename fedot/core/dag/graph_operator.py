from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from fedot.core.dag.graph_node import GraphNode
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence, remove_items
from networkx import graph_edit_distance, set_node_attributes

if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph
    from fedot.core.optimisers.graph import OptGraph


class GraphOperator:
    """_summary_

    :param graph: object used as the :class:`~fedot.core.pipelines.pipeline.Pipeline` structure definition
        or as optimized structure
    :param nodes_postproc_func: nodes postprocessor after their modification
    """

    def __init__(self, graph: Optional[Union['Graph', 'OptGraph']] = None,
                 nodes_postproc_func: Optional[Callable[[List[GraphNode]], None]] = None):
        self._graph = graph
        self._postproc_nodes = nodes_postproc_func or (lambda: ...)

    def delete_node(self, node: GraphNode):
        """Removes provided ``node`` from the bounded graph structure.
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

    def delete_subtree(self, subtree: GraphNode):
        """Deletes given node with all the parents it has, making deletion of the subtree.
        Deletes all edges from removed nodes to remaining graph nodes

        :param subtree: node to be deleted with all of its parents
            and their connections amongst the remaining graph nodes
        """
        subtree_nodes = subtree.ordered_subnodes_hierarchy()
        self._graph._nodes = remove_items(self._graph.nodes, subtree_nodes)
        # prune all edges coming from the removed subtree
        for subtree in self._graph.nodes:
            subtree.nodes_from = remove_items(subtree.nodes_from, subtree_nodes)

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """Replaces ``old_node`` node with ``new_node``

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

    def update_subtree(self, old_subtree: GraphNode, new_subtree: GraphNode):
        """Changes ``old_subtree`` subtree to ``new_subtree``

        :param old_subtree: node and its subtree to be removed
        :param new_subtree: node and its subtree to be placed instead
        """
        new_subtree = deepcopy(new_subtree)
        self.actualise_old_node_children(old_subtree, new_subtree)
        self.delete_subtree(old_subtree)
        self.add_node(new_subtree)
        self.sort_nodes()

    def add_node(self, new_node: GraphNode):
        """Adds new node to the :class:`~fedot.core.pipelines.pipeline.Pipeline` and all of its parent nodes

        :param new_node: new node object to add
        """
        if new_node not in self._graph.nodes:
            self._graph.nodes.append(new_node)
            if new_node.nodes_from:
                for new_parent_node in new_node.nodes_from:
                    self.add_node(new_parent_node)

    def distance_to_root_level(self, node: GraphNode) -> int:
        """Gets distance to the final output node

        :param node: search starting point
        """

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

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        """Gets all the nodes from the chosen layer up to the surface

        :param layer_number: max height of diving

        :return: all nodes from the surface to the ``layer_number`` layer
        """

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

        nodes = get_nodes(self._graph.root_node, current_height=0)
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
        if not isinstance(self._graph.root_node, list):
            self._graph._nodes = self._graph.root_node.ordered_subnodes_hierarchy()

    def node_children(self, node: GraphNode) -> List[GraphNode]:
        """Gets all the ``node`` children

        :param node: for getting children from

        :return: children of the ``node``
        """
        return [other_node for other_node in self._graph.nodes
                if other_node.nodes_from and
                node in other_node.nodes_from]

    def connect_nodes(self, parent: GraphNode, child: GraphNode):
        """Adds edge between ``parent`` and ``child``

        :param parent: acts like parent in pipeline connection relations
        :param child:  acts like child in pipeline connection relations
        """
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
            self._graph.nodes.remove(node)
            if node.nodes_from:
                for node in node.nodes_from:
                    self._clean_up_leftovers(node)

    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         clean_up_leftovers: bool = True):
        """Removes an edge between two nodes

        :param node_parent: where the removing edge comes out
        :param node_child: where the removing edge enters
        :param clean_up_leftovers: whether to remove the remaining invalid vertices with edges or not
        """

        if ((node_child.nodes_from is None or node_parent not in node_child.nodes_from) or
                (node_parent not in self._graph.nodes or node_child not in self._graph.nodes)):
            return
        elif len(node_child.nodes_from) == 1:
            node_child.nodes_from = None
        else:
            node_child.nodes_from.remove(node_parent)

        if clean_up_leftovers:
            self._clean_up_leftovers(node_parent)

        self._postproc_nodes()

    def root_node(self) -> Union[GraphNode, List[GraphNode]]:
        """Gets the final layer node(s) of the graph

        :return: the final layer node(s)
        """
        if not self._graph.nodes:
            return []
        roots = [node for node in self._graph.nodes
                 if not any(self._graph.node_children(node))]
        if len(roots) == 1:
            return roots[0]
        return roots

    def is_graph_equal(self, other_graph: Union['Graph', 'OptGraph']) -> bool:
        """Compares this graph with the ``other_graph``

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
        """Returns graph description

        :return: text graph representation
        """
        return str({
            'depth': self._graph.depth,
            'length': self._graph.length,
            'nodes': self._graph.nodes,
        })

    @property
    def descriptive_id(self) -> str:
        """Returns verbal identificator of the node

        :return: text description of the content in the node and its parameters
        :rtype: str
        """
        root_list = ensure_wrapped_in_sequence(self.root_node())
        full_desc_id = ''.join(r.descriptive_id for r in root_list)
        return full_desc_id

    def graph_depth(self) -> int:
        """Gets this graph depth from its sink-node to its source-node

        :return: length of a path from the root node to the farthest primary node
        """
        if not self._graph.nodes:
            return 0

        def _depth_recursive(node: GraphNode) -> int:
            """Gets this graph depth from the provided ``node`` to the graph source node

            :param node: where to start diving from

            :return: length of a path from the provided ``node`` to the farthest primary node
            """
            if node is None:  # is it real situation to have None in `node.nodes_from`?
                return 0
            if node.nodes_from is None or not node.nodes_from:
                return 1
            else:
                return 1 + max(_depth_recursive(next_node) for next_node in node.nodes_from)

        root = ensure_wrapped_in_sequence(self.root_node())
        return max(_depth_recursive(n) for n in root)

    def get_nodes_degrees(self) -> List[int]:
        """Nodes degree as the number of edges the node has:
            degree = #input_edges + #out_edges

        :return: nodes degrees ordered according to the nx_graph representation of this graph
        """
        graph, _ = graph_structure_as_nx_graph(self._graph)
        index_degree_pairs = graph.degree
        node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
        return node_degrees

    def get_edges(self) -> List[Tuple[GraphNode, GraphNode]]:
        """Gets all available edges in this graph

        :return: pairs of parent_node -> child_node
        """

        edges = []
        for node in self._graph.nodes:
            if node.nodes_from:
                for parent_node in node.nodes_from:
                    edges.append((parent_node, node))
        return edges

def get_distance_between(graph_1: 'Graph', graph_2: 'Graph') -> int:
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
