from core.composer.node import Node
from core.models.data import Data
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model
from typing import (List, Optional)
from abc import abstractmethod
from core.composer.tree_drawing import Tree_Drawing



class GP_NodeGenerator:
    @staticmethod
    def get_primary_node(model: Model, input_data: Data, nodes_to: Optional[Node] = None) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return GP_Primary_Node(input_data=input_data,
                               eval_strategy=eval_strategy, nodes_to=nodes_to)

    @staticmethod
    def get_secondary_node(model: Model, nodes_from: Optional[List['Node']] = None,
                           nodes_to: Optional[Node] = None) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return GP_Secondary_Node(nodes_from=nodes_from,
                                 eval_strategy=eval_strategy, nodes_to=nodes_to)


class GP_Node(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data: Optional[Data],
                 eval_strategy: EvaluationStrategy, nodes_to: Optional[List['Node']]):
        super().__init__(nodes_from=nodes_from,
                         input_data=input_data,
                         eval_strategy=eval_strategy)
        self.nodes_to = nodes_to

    def get_depth_up(self):
        if self.nodes_to:
            depth = self.nodes_to.get_depth_up() + 1
            return depth
        else:
            return 0

    def get_depth_down(self):
        if isinstance(self, GP_Primary_Node):
            return 0
        else:
            return 1 + max([next_node.get_depth_down() for next_node in self.nodes_from])

    def evaluate_branch(self):
        if isinstance(self, GP_Secondary_Node):
            # [node_to.evaluate() for node_to in self.nodes_to]
            self.eval_strategy.model.predict()
        elif isinstance(self, GP_Primary_Node):
            return self.apply()

    def apply(self):
        pass


class GP_Secondary_Node(GP_Node):
    def __init__(self, nodes_from: Optional[Node],
                 eval_strategy: Optional[EvaluationStrategy], nodes_to: Optional[List['Node']]):
        super().__init__(nodes_from=nodes_from,
                         input_data=None,
                         eval_strategy=eval_strategy, nodes_to=nodes_to)
        self.nodes_from = nodes_from
        if not self.nodes_from:
            self.nodes_from = []

    def offspring_fill(self, new_node):
        if new_node:
            self.nodes_from.append(new_node)

    def apply(self):
        pass


class GP_Primary_Node(GP_Node):
    def __init__(self, eval_strategy: Optional[EvaluationStrategy], input_data: Optional[Data],
                 nodes_to: Optional[List['Node']]):
        super().__init__(nodes_from=None,
                         input_data=input_data,
                         eval_strategy=eval_strategy, nodes_to=nodes_to)
