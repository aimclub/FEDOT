from abc import ABC, abstractmethod
from typing import (AnyStr, List, Optional)

import numpy as np

from core.data import Data
from core.evaluation import EvaluationStrategy
from core.model import Model


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 nodes_to: Optional[List['Node']], data_stream,
                 eval_strategy: EvaluationStrategy):
        self.nodes_from = nodes_from
        self.nodes_to = nodes_to
        self.last_parents_ids: Optional[List[AnyStr]]
        self.cached_result = data_stream
        self.eval_strategy = eval_strategy

    @abstractmethod
    def apply(self):
        return self.eval_strategy.evaluate(self.cached_result)


class NodeGenerator:
    def get_primary_mode(self, model: Model):
        eval_strategy = EvaluationStrategy(model=model)
        return PrimaryNode(nodes_to=None, data_stream=None,
                           eval_strategy=eval_strategy)

    def get_secondary_mode(self, model: Model):
        eval_strategy = EvaluationStrategy(model=model)
        return SecondaryNode(nodes_from=None, nodes_to=None, data_stream=None,
                             eval_strategy=eval_strategy)


class PrimaryNode(Node):
    def __init__(self, nodes_to: Optional[List['Node']], data_stream,
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=None,
                         nodes_to=nodes_to,
                         data_stream=data_stream,
                         eval_strategy=eval_strategy)

    def apply(self):
        return self.eval_strategy.evaluate(self.cached_result)


class SecondaryNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self):
        if self.nodes_from is not None:
            parent_predict_list = list()
            for parent in self.nodes_from:
                parent.cached_result = self.cached_result
                parent_predict_list.append(parent.apply())
            parent_predict_list.append(np.arange(0, parent_predict_list[0].size))
            self.cached_result = Data.from_vectors(parent_predict_list)
            return self.eval_strategy.evaluate(self.cached_result)
        return self.eval_strategy.evaluate(self.cached_result)
