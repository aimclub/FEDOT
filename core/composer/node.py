import uuid
from abc import ABC, abstractmethod
from typing import (List, Optional)

import numpy as np

from core.models.data import Data, InputData, OutputData
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data: Optional[Data],
                 eval_strategy: EvaluationStrategy):
        self.node_id = str(uuid.uuid4())
        self.nodes_from = nodes_from
        self.eval_strategy = eval_strategy
        self.input_data = input_data
        self.cached_result = None

    @abstractmethod
    def apply(self) -> OutputData:
        raise NotImplementedError()


class CachedNodeResult:
    def __init__(self, node: Node, model_output: np.array):
        self.cached_output = model_output
        self.last_parents_ids = [n.node_id for n in node.nodes_from] \
            if isinstance(node, SecondaryNode) else None


class NodeGenerator:
    @staticmethod
    def primary_node(model: Model, input_data: Data) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return PrimaryNode(input_data=input_data,
                           eval_strategy=eval_strategy)

    @staticmethod
    def secondary_node(model: Model) -> Node:
        eval_strategy = EvaluationStrategy(model=model)
        return SecondaryNode(nodes_from=None,
                             eval_strategy=eval_strategy)


class PrimaryNode(Node):
    def __init__(self, input_data: Data,
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=None,
                         input_data=input_data,
                         eval_strategy=eval_strategy)
        
    def apply(self) -> OutputData:
        if self.cached_result is not None:
            return OutputData(idx=self.data_stream.idx,
                              features=self.data_stream.features,
                              predict=self.cached_result.cached_output)
        else:
            model_predict = self.eval_strategy.evaluate(self.data_stream)
            self.cached_result = CachedNodeResult(self, model_predict)
            return OutputData(idx=self.data_stream.idx,
                              features=self.data_stream.features,
                              predict=model_predict)


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=nodes_from,
                         input_data=None,
                         eval_strategy=eval_strategy)

    def apply(self) -> OutputData:
        parent_predict_list = list()
        for parent in self.nodes_from:
            parent_predict_list.append(parent.apply())
        target = self.nodes_from[0].data_stream.target
        self.data_stream = Data.from_predictions(outputs=parent_predict_list,
                                                 target=target)
        evaluation_result = self.eval_strategy.evaluate(self.data_stream)
        self.cached_result = CachedNodeResult(self, evaluation_result)
        return OutputData(idx=self.nodes_from[0].data_stream.idx,
                          features=self.nodes_from[0].data_stream.features,
                          predict=evaluation_result)
