from copy import deepcopy
from typing import (Optional)


class GPNode:
    def __init__(self, chain_node, node_to: Optional['GPNode'] = None):
        self.chain_node = chain_node
        self.node_to = node_to

    @property
    def nodes_from(self):
        return self.chain_node.nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes):
        self.chain_node.nodes_from = nodes

    @property
    def eval_strategy(self):
        return self.chain_node.eval_strategy

    @eval_strategy.setter
    def eval_strategy(self, model):
        self.chain_node.eval_strategy.model = model

    @property
    def input_data(self):
        return self.chain_node.input_data

    @property
    def height(self) -> int:
        if self.node_to:
            depth = self.node_to.height + 1
            return depth
        else:
            return 0

    @property
    def depth(self) -> int:
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.depth for next_node in self.nodes_from])

    def nodes_from_height(self, selected_height):
        if self.nodes_from:
            if self.height == selected_height:
                return [self]
            else:
                nodes = []
                for child in self.nodes_from:
                    nodes += child.nodes_from_height(selected_height)
                if nodes:
                    return nodes
                else:
                    return []
        else:
            return []


def swap_nodes(node1, node2):
    new_node = deepcopy(node2)
    new_node.node_to = node1.node_to
    node1.node_to.nodes_from[node1.node_to.nodes_from.index(node1)] = new_node
    node1 = new_node
