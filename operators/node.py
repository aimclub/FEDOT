from typing import List, Optional
from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']], nodes_to: Optional[List['Node']], data_stream):
        self.nodes_from = nodes_from
        self.nodes_to = nodes_to
        self.last_parents_ids: Optional[List['Id']]
        self.EvaluationStrategy = NodeFactory()
        self.cached_result = data_stream


class NodeFactory:
    def fit(self):
        pass

    def predict(self):
        pass


class ModelNode(Node):
    def __init__(self):
        super().__init__()


class OperationNode(Node):
    def __init__(self):
        super().__init__()
