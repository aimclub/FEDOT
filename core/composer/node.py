import collections
import uuid
from abc import ABC, abstractmethod
from typing import (List, Optional)

from core.models.data import Data, InputData, OutputData
from core.models.model import Model
from core.models.model import sklearn_model_by_type
from core.repository.model_types_repository import ModelTypesIdsEnum


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']], model: Model):
        self.node_id = str(uuid.uuid4())
        self.nodes_from = nodes_from
        self.model = model
        self.cached_result = None

    @abstractmethod
    def fit(self, input_data: InputData) -> OutputData:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, input_data: InputData) -> OutputData:
        raise NotImplementedError()

    def __str__(self):
        model = f'{self.model}'
        return model

    def _fit_using_cache(self, input_data):
        if not self._is_cache_actual():
            print('Cache is not actual')
            cached_model, model_predict = self.model.fit(data=input_data)
            self.cached_result = CachedNodeResult(node=self, fitted_model=cached_model)
        else:
            print('Model were obtained from cache')
            model_predict = self.model.predict(fitted_model=self.cached_result.cached_model,
                                               data=input_data)
        return model_predict

    def _is_cache_actual(self):
        return self.cached_result is not None and self.cached_result.is_actual(self)

    @property
    def subtree_nodes(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.subtree_nodes
        return nodes


class CachedNodeResult:
    def __init__(self, node: Node, fitted_model):
        self.cached_model = fitted_model
        self.is_always_actual = isinstance(node, PrimaryNode)
        self.last_parents_ids = [n.node_id for n in node.nodes_from] \
            if isinstance(node, SecondaryNode) else None

    def is_actual(self, node):
        if self.is_always_actual:
            return True
        if not self.last_parents_ids:
            return False
        parent_node_ids = [node.node_id for node in node.nodes_from]
        if not _are_lists_equal(self.last_parents_ids, parent_node_ids):
            return False
        return True


def _are_lists_equal(first, second):
    return collections.Counter(first) == collections.Counter(second)


# TODO: discuss about the usage of NodeGenerator
class NodeGenerator:
    @staticmethod
    def primary_node(model_type: ModelTypesIdsEnum) -> Node:
        return PrimaryNode(model_type=model_type)

    @staticmethod
    def secondary_node(model_type: ModelTypesIdsEnum,
                       nodes_from: Optional[List[Node]] = None) -> Node:
        return SecondaryNode(nodes_from=nodes_from, model_type=model_type)


class PrimaryNode(Node):
    def __init__(self, model_type: ModelTypesIdsEnum):
        model = sklearn_model_by_type(model_type=model_type)
        super().__init__(nodes_from=None, model=model)

    def fit(self, input_data: InputData) -> OutputData:
        print(f'Trying to fit primary node with model: {self.model}')
        model_predict = self._fit_using_cache(input_data=input_data)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict)

    def predict(self, input_data: InputData) -> OutputData:
        print(f'Predict in primary node by model: {self.model}')
        if not self.cached_result:
            raise ValueError('Model must be fitted before predict')

        predict_train = self.model.predict(fitted_model=self.cached_result.cached_model,
                                           data=input_data)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=predict_train)


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 model_type: ModelTypesIdsEnum):
        model = sklearn_model_by_type(model_type=model_type)
        super().__init__(nodes_from=nodes_from, model=model)
        if self.nodes_from is None:
            self.nodes_from = []

    def fit(self, input_data: InputData) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError()
        parent_results = list()
        print(f'Fit all parent nodes in secondary node with model: {self.model}')
        for parent in self.nodes_from:
            parent_results.append(parent.fit(input_data=input_data))

        target = input_data.target
        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        print(f'Trying to fit secondary node with model: {self.model}')

        model_predict = self._fit_using_cache(input_data=secondary_input)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict)

    def predict(self, input_data: InputData) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError('')
        parent_results = list()
        print(f'Obtain predictions from all parent nodes: {self.model}')
        for parent in self.nodes_from:
            parent_results.append(parent.predict(input_data=input_data))

        target = input_data.target
        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        print(f'Obtain prediction in secondary node with model: {self.model}')
        evaluation_result = self.model.predict(fitted_model=self.cached_result.cached_model,
                                               data=secondary_input)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
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
