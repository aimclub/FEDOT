from copy import deepcopy
from typing import (List, Optional)


class GPNode:
    def __init__(self, chain_node, node_to: Optional[List['GPNode']] = None):
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

    def get_height(self):
        if self.node_to:
            depth = self.node_to.get_height() + 1
            return depth
        else:
            return 0

    def get_depth(self) -> int:
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.get_depth() for next_node in self.nodes_from])

    def get_nodes_from_height(self, selected_height):
        if self.nodes_from:
            if self.get_height() == selected_height:
                return [self]
            else:
                nodes = []
                for child in self.nodes_from:
                    nodes += child.get_nodes_from_height(selected_height)
                if nodes:
                    return nodes
                else:
                    return []
        else:
            return []

    def get_similar_nodes(self, other):

        if (self.nodes_from and other.nodes_from) or (not self.nodes_from and not other.nodes_from):
            if self.nodes_from and (len(self.nodes_from) == len(other.nodes_from)):
                nodes = [[self, other]]
                offspring_nodes = []
                for self_c, other_c in zip(self.nodes_from, other.nodes_from):
                    tmp = self_c.get_similar_nodes(other_c)
                    if tmp:
                        if type(tmp[0]) is list:
                            for i in tmp:
                                offspring_nodes.append(i)
                        else:
                            offspring_nodes.append(tmp)
                if offspring_nodes:
                    nodes += offspring_nodes
                return nodes
            elif not self.nodes_from:
                return [self, other]
            else:
                return
        else:
            return


def swap_nodes(node1, node2):
    newnode = deepcopy(node2)
    newnode.node_to = node1.node_to
    node1.node_to.nodes_from[node1.node_to.nodes_from.index(node1)] = newnode
    node1 = newnode
