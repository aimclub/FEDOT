from typing import List, Optional
from abc import ABC
from operators.evaluation import (EvaluationStrategy,
                                  LogRegression,
                                  LinRegression,
                                  XGBoost)


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']], nodes_to: Optional[List['Node']], data_stream,
                 eval_strategy: EvaluationStrategy):
        self.nodes_from = nodes_from
        self.nodes_to = nodes_to
        self.last_parents_ids: Optional[List['Id']]
        self.cached_result = data_stream
        self.eval_strategy = eval_strategy


class NodeFactory:

    def log_reg(self):
        return Node(nodes_from=None, nodes_to=None, data_stream=None, eval_strategy=LogRegression())

    def lin_reg(self):
        return Node(nodes_from=None, nodes_to=None, data_stream=None, eval_strategy=LinRegression())

    def default_xgb(self):
        return Node(nodes_from=None, nodes_to=None, data_stream=None, eval_strategy=XGBoost())

    def nemo(self):
        pass


class ModelNode(Node):
    def __init__(self, nodes_from, nodes_to, data_stream):
        super().__init__(nodes_from, nodes_to, data_stream)


class OperationNode(Node):
    def __init__(self, nodes_from, nodes_to, data_stream):
        super().__init__(nodes_from, nodes_to, data_stream)
