from core.composer.node import Node
from core.models.data import Data
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model
from typing import (List, Optional)

class GP_Primary_Node(Node):
    def __init__(self, nodes_from: Optional[List['Node']],input_data_stream: Data,
                 eval_strategy: EvaluationStrategy, nodes_to: Optional[List['Node']] ):
        super().__init__(nodes_from=nodes_from,
                         input_data_stream=input_data_stream,
                         eval_strategy=eval_strategy)
        self.nodes_to = nodes_to
        if not self.nodes_to:
            self.nodes_to = []

    def offspring_fill (self, new_node):
        if new_node:
            self.nodes_to.append(new_node)

class GP_Secondary_Node(Node):
    def __init__(self, nodes_from: Optional[List['Node']],input_data_stream: Data,
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=nodes_from,
                         input_data_stream=input_data_stream,
                         eval_strategy=eval_strategy)



class NodeGenerator:
    @staticmethod
    def get_primary_node(model: Model, input_data: Data) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return GP_Primary_Node(input_data_stream=input_data,
                               eval_strategy=eval_strategy)
