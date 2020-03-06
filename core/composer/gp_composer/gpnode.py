from copy import deepcopy
from typing import (List, Optional)

from core.composer.node import Node
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.model import Model

class GPNode:
    def __init__(self, chain_node, node_to: Optional[List['Node']]):
        self._chain_node = chain_node
        self.node_to = node_to

    @property
    def nodes_from(self):
        return self._chain_node.nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes):
        self._chain_node.nodes_from = nodes

    @property
    def eval_strategy(self):
        return self._chain_node.eval_strategy

    @eval_strategy.setter
    def eval_strategy(self, model):
        self._chain_node.eval_strategy.model = model

    @property
    def input_data(self):
        return self._chain_node.input_data

    def get_depth_up(self):
        if self.node_to:
            depth = self.node_to.get_depth_up() + 1
            return depth
        else:
            return 0

    def get_depth_down(self):
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.get_depth_down() for next_node in self.nodes_from])

    def get_nodes_from_layer(self, selected_depth):
        if self.nodes_from:
            if self.get_depth_up() == selected_depth:
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

    def swap_nodes(self, other):
        newnode = deepcopy(other)
        newnode.node_to = self.node_to
        self.node_to.nodes_from[self.node_to.nodes_from.index(self)] = newnode
        self = newnode


class GPNodeGenerator:
    @staticmethod
    def primary_node(model: Model, input_data: InputData, node_to: Optional[Node] = None, ) -> GPNode:
        chain_node = NodeGenerator.primary_node(model=model, input_data=input_data)
        return GPNode(chain_node=chain_node, node_to=node_to)

    @staticmethod
    def secondary_node(model: Model, node_to: Optional[Node] = None, nodes_from: Optional[List['Node']] =None ) -> GPNode:
        chain_node = NodeGenerator.secondary_node(model=model, nodes_from= nodes_from)
        return GPNode(chain_node=chain_node, node_to=node_to)
