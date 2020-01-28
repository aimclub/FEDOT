from typing import List, Optional
from abc import ABC, abstractmethod
from operators.evaluation import EvaluationStrategy, LogRegression, LinRegression


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']], nodes_to: Optional[List['Node']], data_stream):
        self.nodes_from = nodes_from
        self.nodes_to = nodes_to
        self.last_parents_ids: Optional[List['Id']]
        self.cached_result = data_stream
        self.evaluation_strategy = EvaluationStrategy


class NodeFactory:
    def __init__(self, model):
        self.model = model

    def log_reg(self):
        self.model.evaluation_strategy = LogRegression()
        return self.model

    def lin_reg(self):
        self.model.evaluation_strategy = LinRegression()
        return self.model

    def default_xgb(self):
        pass

    def nemo(self):
        pass


class ModelNode(Node):
    def __init__(self, nodes_from, nodes_to, data_stream):
        super().__init__(nodes_from, nodes_to, data_stream)


class OperationNode(Node):
    def __init__(self, nodes_from, nodes_to, data_stream):
        super().__init__(nodes_from, nodes_to, data_stream)
