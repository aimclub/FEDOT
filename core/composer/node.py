import uuid
from abc import ABC, abstractmethod
from typing import (List, Optional)

import numpy as np

from core.models.data import Data, InputData, OutputData
from core.models.model import Model
from core.models.model import sklearn_model_by_type
from core.repository.model_types_repository import ModelTypesIdsEnum


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 input_data: Optional[InputData],
                 model: Model):
        self.node_id = str(uuid.uuid4())
        self.nodes_from = nodes_from
        self.model = model
        self.input_data = input_data
        self.cached_result = None
        self.is_caching = True

    @abstractmethod
    def apply(self) -> OutputData:
        raise NotImplementedError()

    def __str__(self):
        model = f'{self.model}'
        return model

    @property
    def subtree_nodes(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.subtree_nodes
        return nodes


class CachedNodeResult:
    def __init__(self, node: Node, model_output: np.array):
        self.cached_output = model_output
        self.is_always_actual = isinstance(node, PrimaryNode)
        self.last_parents_ids = [n.node_id for n in node.nodes_from] \
            if isinstance(node, SecondaryNode) else None

    # @property
    # def data(self, node: Node):
    #     if self.last_parents_ids not in node:#TODO
    #         return None
    #     else:
    #         self.cached_output
    #
    # @property
    # def model(self, node: Node) -> Model:
    #     if self.last_parents_ids not in node:  # TODO
    #         return None
    #     else:
    #         self.model
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


# TODO: discuss about the usage of NodeGenerator
class NodeGenerator:
    @staticmethod
    def primary_node(model_type: ModelTypesIdsEnum,
                     input_data: Optional[InputData]) -> Node:
        return PrimaryNode(model_type=model_type, input_data=input_data)

    @staticmethod
    def secondary_node(model_type: ModelTypesIdsEnum,
                       nodes_from: Optional[List[Node]] = None) -> Node:
        return SecondaryNode(nodes_from=nodes_from, model_type=model_type)


class PrimaryNode(Node):
    def __init__(self, model_type: ModelTypesIdsEnum, input_data: InputData):
        model = sklearn_model_by_type(model_type=model_type)
        super().__init__(nodes_from=None, input_data=input_data, model=model)

    def apply(self) -> OutputData:
        if self.is_caching and self.cached_result is not None and self.cached_result.is_actual(self.nodes_from):
            return OutputData(idx=self.input_data.idx,
                              features=self.input_data.features,
                              predict=self.cached_result.cached_output)
        else:
            model_predict = self.model.evaluate(self.input_data)
            if self.is_caching:
                self.cached_result = CachedNodeResult(self, model_predict)
            return OutputData(idx=self.input_data.idx,
                              features=self.input_data.features,
                              predict=model_predict)


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 model_type: ModelTypesIdsEnum):
        model = sklearn_model_by_type(model_type=model_type)
        super().__init__(nodes_from=nodes_from, input_data=None, model=model)

        if self.nodes_from is None:
            self.nodes_from = []

    def apply(self) -> OutputData:
        parent_predict_list = list()
        for parent in self.nodes_from:
            parent_predict_list.append(parent.apply())
        if len(self.nodes_from) == 0:
            raise ValueError
        target = self.nodes_from[0].input_data.target
        self.input_data = Data.from_predictions(outputs=parent_predict_list,
                                                target=target)
        evaluation_result = self.model.evaluate(self.input_data)
        if self.is_caching:
            self.cached_result = CachedNodeResult(self, evaluation_result)
        return OutputData(idx=self.nodes_from[0].input_data.idx,
                          features=self.nodes_from[0].input_data.features,
                          predict=evaluation_result)


def equivalent_subtree(root_of_tree_first: Node, root_of_tree_second: Node) -> List[Tuple[Any, Any]]:
    """returns the nodes set of the structurally equivalent subtree as: list of pairs [node_from_tree1, node_from_tree2]
    where: node_from_tree1 and node_from_tree2 are equivalent nodes from tree1 and tree2 respectively"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and (isinstance(node_first, PrimaryNode) or len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(root_of_tree_first, root_of_tree_second)
    assert isinstance(pairs_set, list)
    return pairs_set
