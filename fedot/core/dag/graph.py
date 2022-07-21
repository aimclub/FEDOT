from typing import Any, List, Optional, Tuple, Union

from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.visualisation.graph_viz import GraphVisualiser


class Graph:
    """
    Base class used for the :class:`~fedot.core.pipelines.pipeline.Pipeline` structure definition

    :param nodes: pipeline nodes
    """

    def __init__(self, nodes: Optional[Union['GraphNode', List['GraphNode']]] = None):
        self._nodes: List[GraphNode] = []
        self._operator = GraphOperator(self, self._empty_postproc)

        if nodes:
            for node in ensure_wrapped_in_sequence(nodes):
                self.add_node(node)

    def _empty_postproc(self, nodes=None):  # TODO: maybe it should return nodes as is instead?
        """
        Doesn't do any postprocessing to the provided ``nodes``

        :param nodes: _description_
        """
        pass

    @property
    def nodes(self):
        return self._nodes

    def add_node(self, new_node: GraphNode):
        """
        Adds new node to the :class:`~fedot.core.pipelines.pipeline.Pipeline`

        :param new_node: node to be added
        """
        self._operator.add_node(new_node)

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """
        Replaces ``old_node`` with ``new_node``

        :param old_node: node to be replaced
        :param new_node: node to be placed instead
        """

        self._operator.update_node(old_node, new_node)

    def delete_node(self, node: 'GraphNode'):
        """
        Deletes provided ``node`` and redirects all of its parents to its child.

        :param node: to be deleted
        """

        self._operator.delete_node(node)

    def update_subtree(self, old_subtree: 'GraphNode', new_subtree: 'GraphNode'):
        """
        Replaces ``old_subtree`` with ``new_subtree``

        :param old_subtree: to be replaced
        :param new_subtree: to be placed instead
        """
        self._operator.update_subtree(old_subtree, new_subtree)

    def delete_subtree(self, subroot: GraphNode):
        """
        Deletes given subtree with node as subroot.

        :param subroot: to be deleted
        """
        self._operator.delete_subtree(subroot)

    def distance_to_root_level(self, node: GraphNode) -> int:
        """ Returns distance to root level """
        return self._operator.distance_to_root_level(node=node)

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        """ Returns all nodes from specified layer """
        return self._operator.nodes_from_layer(layer_number=layer_number)

    def node_children(self, node: GraphNode) -> List[Optional[GraphNode]]:
        """ Returns all node's children """
        return self._operator.node_children(node=node)

    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        """ Add an edge from node_parent to node_child """
        self._operator.connect_nodes(parent=node_parent, child=node_child)

    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         is_clean_up_leftovers: bool = True):
        """ Delete an edge from node_parent to node_child """
        self._operator.disconnect_nodes(node_parent=node_parent, node_child=node_child,
                                        is_clean_up_leftovers=is_clean_up_leftovers)

    def get_nodes_degrees(self):
        """ Nodes degree as the number of edges the node has:
         k = k(in) + k(out) """
        return self._operator.get_nodes_degrees()

    def get_edges(self) -> List[Tuple[GraphNode, GraphNode]]:
        """ Returns all available edges in a given graph """
        return self._operator.get_edges()

    def show(self, path: Optional[str] = None):
        """
        Visualizes graph or saves its picture to the specified ``path``

        :param path: optional, save location of the graph visualization image
        """
        GraphVisualiser().visualise(self, path)

    def __eq__(self, other: 'Graph') -> bool:
        """
        Compares this graph with the ``other``

        :param other: another graph

        :return: is it equal to ``other`` in terms of the graphs
        """
        return self._operator.is_graph_equal(other)

    def __str__(self) -> str:
        """
        Returns graph description

        :return: text graph representation
        """
        return self._operator.graph_description()

    def __repr__(self) -> str:
        """
        Does the same as :meth:`__str__`

        :return: text graph representation
        """
        return self.__str__()

    @property
    def root_node(self) -> Union[GraphNode, List[GraphNode]]:
        """
        Finds all the sink-nodes of the graph

        :return: the final predictors-nodes
        """
        roots = self._operator.root_node()
        return roots

    @property
    def descriptive_id(self):
        return self._operator.descriptive_id

    @property
    def length(self) -> int:
        """
        Returns size of the graph


        :return: number of nodes in the graph
        """
        return len(self.nodes)

    @property
    def depth(self) -> int:
        """
        Returns depth of the graph starting from the farthest root node

        :return: depth of the graph
        """
        return self._operator.graph_depth()
