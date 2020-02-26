
from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import Data


class Chain:
    def __init__(self, base_node: Optional[Node] = None):
        if base_node is None:
            self.nodes = []
        else:
            self.nodes = self._flat_nodes_tree(base_node)

    def evaluate(self, new_data: Optional[Data] = None) -> Data:
        if new_data is not None:
            # if the chain should be evaluated for the new dataset
            for node in self.nodes:
                if isinstance(node, PrimaryNode):
                    node.input_data = new_data
                node.cached_result = None
                # TODO clean cache and choice strategy for trained models
        return self.root_node.apply()

    def add_node(self, new_node: Node):
        # Append new node to chain
        self.nodes.append(new_node)

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    def _is_node_has_child(self, node):
        return any([(node in other_node.nodes_from)
                    for other_node in self.nodes if isinstance(other_node, SecondaryNode)])

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not self._is_node_has_child(node)][0]
        return root

    @property
    def length(self):
        return len(self.nodes)

    @property
    def depth(self):
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)

    def _flat_nodes_tree(self, node):
        raise NotImplementedError()
