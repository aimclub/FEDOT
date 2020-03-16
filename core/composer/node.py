import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (List, Optional)

import numpy as np

from core.models.data import Data, InputData, OutputData
from core.models.evaluation import EvaluationStrategy
from core.models.model import Model


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data: Optional[InputData],
                 eval_strategy: EvaluationStrategy):
        self.node_id = str(uuid.uuid4())
        self.nodes_from = nodes_from
        self.eval_strategy = eval_strategy
        self.input_data = input_data
        self.cached_result = None
        self.is_caching = True

    @abstractmethod
    def apply(self) -> OutputData:
        raise NotImplementedError()

    def __str__(self):
        model = f'{self.eval_strategy.model}'
        return model

    def get_similar_nodes(self, other):

        if (self.nodes_from and other.nodes_from) or (not self.nodes_from and not other.nodes_from):
            if self.nodes_from and (len(self.nodes_from) == len(other.nodes_from)):
                nodes = [[self, other]]
                offspring_nodes = []
                for self_c, other_c in zip(self.nodes_from, other.nodes_from):
                    tmp = self_c.get_similar_nodes(other_c)
                    if tmp:
                        if type(tmp[0]) is list:
                            for i in tmp:
                                offspring_nodes.append(i)
                        else:
                            offspring_nodes.append(tmp)
                if offspring_nodes:
                    nodes += offspring_nodes
                return nodes
            elif not self.nodes_from:
                return [self, other]
            else:
                return
        else:
            return

class CachedNodeResult:
    def __init__(self, node: Node, model_output: np.array):
        self.cached_output = model_output
        self.is_always_actual = isinstance(node, PrimaryNode)
        self.last_parents_ids = [n.node_id for n in node.nodes_from] \
            if isinstance(node, SecondaryNode) else None

    def is_actual(self, parent_nodes):
        if self.is_always_actual:
            return True
        if not self.last_parents_ids or self.last_parents_ids is None:
            return False
        if len(self.last_parents_ids) != len(parent_nodes):
            return False
        for id in self.last_parents_ids:
            if id not in [node.node_id for node in parent_nodes]:
                return False
        return True


class NodeGenerator:
    @staticmethod
    def primary_node(model: Model, input_data: Optional[InputData]) -> Node:
        eval_strategy = EvaluationStrategy(model=deepcopy(model))
        return PrimaryNode(input_data=input_data,
                           eval_strategy=eval_strategy)

    @staticmethod
    def secondary_node(model: Model, nodes_from: Optional[List[Node]] = None) -> Node:
        eval_strategy = EvaluationStrategy(model=deepcopy(model))
        return SecondaryNode(nodes_from=nodes_from,
                             eval_strategy=eval_strategy)


class PrimaryNode(Node):
    def __init__(self, input_data: InputData,
                 eval_strategy: EvaluationStrategy):
        super().__init__(nodes_from=None,
                         input_data=input_data,
                         eval_strategy=eval_strategy)

    def apply(self) -> OutputData:
        if self.is_caching and self.cached_result is not None and self.cached_result.is_actual(self.nodes_from):
            return OutputData(idx=self.input_data.idx,
                              features=self.input_data.features,
                              predict=self.cached_result.cached_output)
        else:
            model_predict = self.eval_strategy.evaluate(self.input_data)
            if self.is_caching:
                self.cached_result = CachedNodeResult(self, model_predict)
            return OutputData(idx=self.input_data.idx,
                              features=self.input_data.features,
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
        if len(self.nodes_from) == 0:
            raise ValueError
        target = self.nodes_from[0].input_data.target
        self.input_data = Data.from_predictions(outputs=parent_predict_list,
                                                target=target)
        evaluation_result = self.eval_strategy.evaluate(self.input_data)
        if self.is_caching:
            self.cached_result = CachedNodeResult(self, evaluation_result)
        return OutputData(idx=self.nodes_from[0].input_data.idx,
                          features=self.nodes_from[0].input_data.features,
                          predict=evaluation_result)
