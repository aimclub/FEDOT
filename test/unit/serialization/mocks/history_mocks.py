from typing import Union, List, Sequence

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode


class CustomMockNode(GraphNode):
    def __init__(self, content: dict = None, nodes_from: list = None):
        self._content = content
        self._nodes_from = nodes_from or []

    @property
    def nodes_from(self):
        return self._nodes_from

    def __str__(self):
        return self._content['name']


class CustomMockGraph(Graph):
    def __init__(self, nodes: Union[CustomMockNode, List[CustomMockNode]] = ()):
        self._nodes = nodes

    def add_node(self, node: CustomMockNode):
        pass

    def update_node(self, old_node: CustomMockNode, new_node: CustomMockNode):
        pass

    def update_subtree(self, old_subtree: CustomMockNode, new_subtree: CustomMockNode):
        pass

    def delete_node(self, node: CustomMockNode):
        pass

    def delete_subtree(self, subroot: CustomMockNode):
        pass

    def node_children(self, node: CustomMockNode):
        pass

    def connect_nodes(self, node_parent: CustomMockNode, node_child: CustomMockNode):
        pass

    def disconnect_nodes(self, node_parent: CustomMockNode, node_child: CustomMockNode,
                         clean_up_leftovers: bool = True):
        pass

    def get_edges(self):
        pass

    def __eq__(self, other_graph: 'CustomMockGraph'):
        pass

    def root_nodes(self) -> Sequence[CustomMockNode]:
        return [self._nodes[0]]

    @property
    def nodes(self) -> List[CustomMockNode]:
        return self._nodes

    @property
    def depth(self) -> int:
        pass
