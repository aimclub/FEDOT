import uuid
from abc import ABC
from typing import (List, Optional)

from core.data import Data
from core.evaluation import EvaluationStrategy
from core.model import Model


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data_stream: Optional[Data],
                 eval_strategy: EvaluationStrategy):
        self.node_id = str(uuid.uuid4())
        self.nodes_from = nodes_from
        self.eval_strategy = eval_strategy
        self.data_stream = input_data_stream
        self.cached_result = None

    def apply(self) -> Data:
        return self.eval_strategy.evaluate(self.data_stream)


class CachedResult:
    def __init__(self, node: Node, model_output: Data):
        self.cached_output = model_output
        self.last_parents_ids = [n.node_id for n in node.nodes_from] \
            if isinstance(node, SecondaryNode) else None


class NodeGenerator:
    def get_primary_mode(self, model: Model, input_data: Data) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return PrimaryNode(input_data_stream=input_data,
                           eval_strategy=eval_strategy)

    def get_secondary_mode(self, model: Model) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return SecondaryNode(nodes_from=None,
                             eval_strategy=eval_strategy)


class PrimaryNode(Node):
    def __init__(self, input_data_stream: Data,
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=None,
                         input_data_stream=input_data_stream,
                         eval_strategy=eval_strategy)


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=nodes_from,
                         input_data_stream=None,
                         eval_strategy=eval_strategy)

    def apply(self) -> Data:
        evaluation_result = self.eval_strategy.evaluate(self.cached_result.cached_output)
        self.cached_result = CachedResult(self, evaluation_result)
        return evaluation_result
