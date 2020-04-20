from abc import ABC, abstractmethod
from copy import copy
from typing import (List, Optional, Any, Tuple)

from core.models.data import Data, InputData, OutputData
from core.models.model import Model
from core.repository.model_types_repository import ModelTypesIdsEnum


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']], model: Model):
        self.nodes_from = nodes_from
        self.model = model
        self.cache = FittedModelCache(self)

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        node_label = self.model.description
        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()  # models with different inputs order are equal
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    @abstractmethod
    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, input_data: InputData, verbose=False) -> OutputData:
        raise NotImplementedError()

    def __str__(self):
        model = f'{self.model}'
        return model

    def _fit_using_cache(self, input_data, verbose=False):
        self.model.init(task=input_data.task_type)
        if not self.cache.actual_cached_model:
            if verbose:
                print('Cache is not actual')
            cached_model, model_predict = self.model.fit(data=input_data)
            self.cache.append(cached_model)
        else:
            if verbose:
                print('Model were obtained from cache')
            model_predict = self.model.predict(fitted_model=self.cache.actual_cached_model,
                                               data=input_data)
        return model_predict

    @property
    def subtree_nodes(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.subtree_nodes
        return nodes


class FittedModelCache:
    def __init__(self, related_node: Node):
        self._local_cached_models = {}
        self._related_node_ref = related_node
        self.global_cached_models = None

    def append(self, fitted_model):
        self._local_cached_models[self._related_node_ref.descriptive_id] = fitted_model
        if self.global_cached_models is not None:
            self.global_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    def import_from_other_cache(self, other_cache: 'FittedModelCache'):
        for entry_key in other_cache._local_cached_models.keys():
            self._local_cached_models[entry_key] = other_cache._local_cached_models[entry_key]

    def clear(self):
        self._local_cached_models = {}
        self.global_cached_models = None

    @property
    def actual_cached_model(self):
        found_model = self._local_cached_models.get(self._related_node_ref.descriptive_id, None)
        if not found_model and self.global_cached_models:
            found_model = self.global_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


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
        model = Model(model_type=model_type)
        super().__init__(nodes_from=None, model=model)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        if verbose:
            print(f'Trying to fit primary node with model: {self.model}')
        model_predict = self._fit_using_cache(input_data=input_data, verbose=verbose)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict, task_type=input_data.task_type)

    def predict(self, input_data: InputData, verbose=False) -> OutputData:
        if verbose:
            print(f'Predict in primary node by model: {self.model}')
        if not self.cache:
            raise ValueError('Model must be fitted before predict')

        predict_train = self.model.predict(fitted_model=self.cache.actual_cached_model,
                                           data=input_data)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=predict_train, task_type=input_data.task_type)


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 model_type: ModelTypesIdsEnum):
        model = Model(model_type=model_type)
        super().__init__(nodes_from=nodes_from, model=model)
        if self.nodes_from is None:
            self.nodes_from = []

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError()
        parent_results = []

        if verbose:
            print(f'Fit all parent nodes in secondary node with model: {self.model}')
        for parent in self.nodes_from:
            parent_results.append(parent.fit(input_data=input_data))

        target = input_data.target
        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        if verbose:
            print(f'Trying to fit secondary node with model: {self.model}')

        model_predict = self._fit_using_cache(input_data=secondary_input)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict,
                          task_type=input_data.task_type)

    def predict(self, input_data: InputData, verbose=False) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError('')
        parent_results = []
        if verbose:
            print(f'Obtain predictions from all parent nodes: {self.model}')
        for parent in self.nodes_from:
            parent_results.append(parent.predict(input_data=input_data))

        target = input_data.target
        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        if verbose:
            print(f'Obtain prediction in secondary node with model: {self.model}')
        evaluation_result = self.model.predict(fitted_model=self.cache.actual_cached_model,
                                               data=secondary_input)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=evaluation_result,
                          task_type=input_data.task_type)


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
