from core.composer.node import Node
from core.models.data import InputData
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model
from typing import (List, Optional)
from abc import abstractmethod
from core.composer.node import NodeGenerator


class GP_NodeGenerator:
    @staticmethod
    def primary_node(model: Model, input_data: InputData, nodes_to: Optional[Node] = None) -> Node:
        chain_node = NodeGenerator.primary_node(model=model, input_data=input_data)
        return GP_Node(chain_node=chain_node, nodes_to=nodes_to)

    @staticmethod
    def secondary_node(model: Model, nodes_to: Optional[Node] = None) -> Node:
        chain_node = NodeGenerator.secondary_node(model=model)
        return GP_Node(chain_node=chain_node, nodes_to=nodes_to)


class GP_Node:
    def __init__(self, chain_node, nodes_to: Optional[List['Node']]):
        self.__chain_node = chain_node
        self.nodes_to = nodes_to

    @property
    def nodes_from(self):
        return self.__chain_node.nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes):
        self.__chain_node.nodes_from = nodes

    @property
    def eval_strategy(self):
        return self.__chain_node.eval_strategy

    def get_depth_up(self):
        if self.nodes_to:
            depth = self.nodes_to.get_depth_up() + 1
            return depth
        else:
            return 0

    def get_depth_down(self):
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.get_depth_down() for next_node in self.nodes_from])
