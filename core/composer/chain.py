from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import Data
from core.composer.gp_composer.gp_node import GP_NodeGenerator
from random import randint, choice


class Chain:
    def __init__(self, ):
        self.nodes = []
        self.root = None

    def evaluate(self) -> Data:
        return self.root_node.apply()

    def tree_evaluation(self)->Data:
        return self.root.evaluate_branch()

    def evaluate_with_specific_data(self, new_data: Data) -> Data:
        for node in self.nodes:
            if isinstance(node, PrimaryNode):
                node.data_stream = new_data
        return self.root_node.apply()

    def add_node(self, new_node: Node):
        # Append new node to chain
        self.nodes.append(new_node)

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not any([(node in other_node.nodes_from)
                            for other_node in self.nodes if isinstance(other_node, SecondaryNode)])][0]
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

    def get_depth_up(self):
        return self.root.get_depth_up()

    def get_depth_down(self):
        return self.root.get_depth_down()
