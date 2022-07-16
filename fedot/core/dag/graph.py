from typing import TYPE_CHECKING, List, Optional, Union, Any

from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.visualisation.graph_viz import GraphVisualiser

if TYPE_CHECKING:
    from fedot.core.dag.graph_node import GraphNode

from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence


class Graph:
    """
    Base class used for the pipeline structure definition

    :param nodes: 'GraphNode' object(s)
    """

    def __init__(self, nodes: Optional[Union['GraphNode', List['GraphNode']]] = None):
        self.nodes = []
        self.operator = GraphOperator(self, self._empty_postproc)

        if nodes:
            for node in ensure_wrapped_in_sequence(nodes):
                self.add_node(node)

    def _empty_postproc(self, nodes=None):
        pass

    def add_node(self, new_node: 'GraphNode'):
        """
        Add new node to the Pipeline

        :param new_node: new GraphNode object
        """
        self.operator.add_node(new_node)

    def update_node(self, old_node: 'GraphNode', new_node: 'GraphNode'):
        """
        Replace old_node with new one.

        :param old_node: 'GraphNode' object to replace
        :param new_node: 'GraphNode' new object
        """

        self.operator.update_node(old_node, new_node)

    def delete_node(self, node: 'GraphNode'):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: 'GraphNode' object to delete
        """

        self.operator.delete_node(node)

    def update_subtree(self, old_subroot: 'GraphNode', new_subroot: 'GraphNode'):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: 'GraphNode' object to replace
        :param new_subroot: 'GraphNode' new object
        """
        self.operator.update_subtree(old_subroot, new_subroot)

    def delete_subtree(self, subroot: 'GraphNode'):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        self.operator.delete_subtree(subroot)

    def distance_to_root_level(self, node: 'GraphNode') -> int:
        return self.operator.distance_to_root_level(node=node)

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        return self.operator.nodes_from_layer(layer_number=layer_number)

    def show(self, path: str = None):
        GraphVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        return self.operator.is_graph_equal(other)

    def __str__(self):
        return self.operator.graph_description()

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self):
        roots = self.operator.root_node()
        return roots

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        return self.operator.graph_depth()
