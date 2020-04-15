from copy import deepcopy
from typing import (Optional, List, Any)
from core.composer.node import node_duplicate


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
    def node_id(self):
        return self.chain_node.node_id

    @node_id.setter
    def node_id(self, id):
        self.chain_node.node_id = id

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

    @property
    def duplicate(self) -> 'GPNode':
        return node_duplicate(self)

    def nodes_from_height(self, selected_height) -> List[Any]:
        nodes = []
        if self.nodes_from:
            if self.height == selected_height:
                nodes.append(self)
            else:
                for child in self.nodes_from:
                    nodes += child.nodes_from_height(selected_height)
        return nodes


def swap_nodes(node_first, node_second):
    new_node = node_second.duplicate
    new_node.node_to = node_first.node_to
    node_first.node_to.nodes_from[node_first.node_to.nodes_from.index(node_first)] = new_node
    node_first = new_node
