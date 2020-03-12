from copy import deepcopy
from typing import (List, Optional)
from core.composer.node import Node
import enum


class GPNode:
    def __init__(self, chain_node, node_to: Optional[List['Node']] = None):
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

    def get_depth_to_final(self):
        if self.node_to:
            depth = self.node_to.get_depth_to_final() + 1
            return depth
        else:
            return 0

    def get_depth_to_primary(self) -> int:
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.get_depth_to_primary() for next_node in self.nodes_from])

    def get_nodes_from_layer(self, selected_depth):
        if self.nodes_from:
            if self.get_depth_to_final() == selected_depth:
                return [self]
            else:
                nodes = []
                for child in self.nodes_from:
                    nodes += child.get_nodes_from_layer(selected_depth)
                if nodes:
                    return nodes
                else:
                    return []
        else:
            return []


def swap_nodes(node1, node2):
    newnode = deepcopy(node2)
    newnode.node_to = node1.node_to
    node1.node_to.nodes_from[node1.node_to.nodes_from.index(node1)] = newnode
    node1 = newnode
