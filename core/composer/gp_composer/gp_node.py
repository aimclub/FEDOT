from core.composer.node import Node
from core.models.data import Data
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model
from typing import (List, Optional)
from abc import abstractmethod

class GP_NodeGenerator:
    @staticmethod
    def get_primary_node(model: Model, input_data: Data,  nodes_from: Optional[List['Node']]) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return GP_Primary_Node(input_data_stream=input_data,
                           eval_strategy=eval_strategy, nodes_from=nodes_from)

    @staticmethod
    def get_secondary_node(model: Model, nodes_from: Optional[List['Node']] = None, nodes_to: Optional[List['Node']] =None ) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return GP_Secondary_Node(nodes_from=nodes_from,
                             eval_strategy=eval_strategy, nodes_to= nodes_to)
class GP_Node(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data_stream: Optional[Data],
                 eval_strategy: EvaluationStrategy):
            super().__init__(nodes_from=nodes_from,
                             input_data_stream=input_data_stream,
                             eval_strategy=eval_strategy)

    @abstractmethod
    def apply(self) -> Data:
        raise NotImplementedError()


class GP_Secondary_Node(GP_Node):
    def __init__(self, nodes_from: Optional[Node] ,
                 eval_strategy: EvaluationStrategy, nodes_to: Optional[List['Node']]):
        super().__init__(nodes_from=nodes_from,
                         input_data_stream=None,
                         eval_strategy=eval_strategy)
        self.nodes_to = nodes_to
        if not self.nodes_to:
            self.nodes_to = []

    def offspring_fill (self, new_node):
        if new_node:
            self.nodes_to.append(new_node)

    def apply(self) -> Data:
        return self.eval_strategy.evaluate(self.data_stream)

class GP_Primary_Node(GP_Node):
    def __init__(self, nodes_from: Optional[Node],
                 eval_strategy: EvaluationStrategy,input_data_stream: Data):
        super().__init__(nodes_from=nodes_from,
                         input_data_stream=input_data_stream,
                         eval_strategy=eval_strategy)

    def apply(self) -> Data:
        return self.eval_strategy.evaluate(self.data_stream)
