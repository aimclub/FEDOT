from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.utils import copy_doc
from fedot.core.visualisation.graph_viz import GraphVisualiser

if TYPE_CHECKING:
    from fedot.core.dag.graph_node import GraphNode



class Graph:
    """Base class used for the :obj:`Pipeline` structure definition

    Args:
        nodes: pipeline nodes
    """

    def __init__(self, nodes: Optional[Union['GraphNode', List['GraphNode']]] = None):
        self._nodes: List['GraphNode'] = []
        self._operator = GraphOperator(self, self._empty_postproc)

        if nodes:
            for node in ensure_wrapped_in_sequence(nodes):
                self.add_node(node)

    def _empty_postproc(self,
                        nodes: Optional[List['GraphNode']] = None):  # TODO: maybe it should return nodes as is instead?
        """Does not do any postprocessing to the provided ``nodes``

        Args:
            nodes: not obligatory
        """

        pass

    @property
    def nodes(self):
        return self._nodes

    @copy_doc(GraphOperator.add_node)
    def add_node(self, new_node: 'GraphNode'):
        self._operator.add_node(new_node)

    @copy_doc(GraphOperator.update_node)
    def update_node(self, old_node: 'GraphNode', new_node: 'GraphNode'):
        self._operator.update_node(old_node, new_node)

    @copy_doc(GraphOperator.delete_node)
    def delete_node(self, node: 'GraphNode'):
        self._operator.delete_node(node)

    @copy_doc(GraphOperator.update_subtree)
    def update_subtree(self, old_subtree: 'GraphNode', new_subtree: 'GraphNode'):
        self._operator.update_subtree(old_subtree, new_subtree)

    @copy_doc(GraphOperator.delete_subtree)
    def delete_subtree(self, subtree: 'GraphNode'):
        self._operator.delete_subtree(subtree)

    def distance_to_root_level(self, node: GraphNode) -> int:
        """ Returns distance to root level
        """

        return self._operator.distance_to_root_level(node=node)

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        """ Returns all nodes from specified layer 
        """

        return self._operator.nodes_from_layer(layer_number=layer_number)

    def node_children(self, node: GraphNode) -> List[Optional[GraphNode]]:
        """Returns all node's children
        """

        return self._operator.node_children(node=node)

    def connect_nodes(self, node_parent: GraphNode, node_child: GraphNode):
        """Add an edge from node_parent to node_child
        """

        self._operator.connect_nodes(parent=node_parent, child=node_child)

    def disconnect_nodes(self, node_parent: GraphNode, node_child: GraphNode,
                         clean_up_leftovers: bool = True):
        """Delete an edge from node_parent to node_child
        """

        self._operator.disconnect_nodes(node_parent=node_parent, node_child=node_child,
                                        clean_up_leftovers=clean_up_leftovers)

    def get_nodes_degrees(self):
        """Nodes degree as the number of edges the node has: 
        ``k = k(in) + k(out)``
        """

        return self._operator.get_nodes_degrees()

    def get_edges(self) -> List[Tuple[GraphNode, GraphNode]]:
        """Returns all available edges in a given graph
        """

        return self._operator.get_edges()

    def show(self, path: Optional[str] = None):
        """Visualizes graph or saves its picture to the specified ``path``

        Args:
            path: save location of the graph visualization image
        """

        GraphVisualiser().visualise(self, path)

    def __eq__(self, other_graph: 'Graph') -> bool:
        """Compares this graph with the ``other_graph``

        Args:
            other_graph: another graph

        Returns:
            is it equal to ``other_graph`` in terms of the graphs
        """

        return self._operator.is_graph_equal(other_graph)

    @copy_doc(GraphOperator.graph_description)
    def __str__(self) -> str:
        return self._operator.graph_description()

    def __repr__(self) -> str:
        """Does the same as :meth:`__str__`

        Returns:
            text graph representation
        """

        return self.__str__()

    @property
    def root_node(self) -> Union['GraphNode', List['GraphNode']]:
        """Finds all the sink-nodes of the graph

        Returns:
            the final predictors-nodes
        """

        roots = self._operator.root_node()
        return roots

    @property
    def descriptive_id(self):
        return self._operator.descriptive_id

    @property
    def length(self) -> int:
        """Returns size of the graph

        Returns:
            number of nodes in the graph
        """

        return len(self.nodes)

    @property
    def depth(self) -> int:
        """Returns depth of the graph starting from the farthest root node

        Returns:
            depth of the graph
        """

        return self._operator.graph_depth()
