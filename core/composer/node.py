from abc import ABC, abstractmethod
from collections import namedtuple
from copy import copy
from typing import (List, Optional)

from core.models.data import Data, InputData, OutputData
from core.models.model import Model
from core.models.preprocessing import *
from core.models.transformation import transformation_function_for_data
from core.repository.dataset_types import DataTypesEnum

CachedState = namedtuple('CachedState', 'preprocessor model')


class Node(ABC):

    def __init__(self, nodes_from: Optional[List['Node']], model: Model):
        self.nodes_from = nodes_from
        self.model = model
        self.cache = FittedModelCache(self)
        self.label = []

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
            previous_items.sort()
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

    def _fit_using_cache(self, input_data, with_preprocessing=True, verbose=False):

        preprocessed_data = transformation_function_for_data(
            input_data_type=input_data.data_type,
            required_data_types=self.model.metadata.input_types)(input_data)

        if not self.cache.actual_cached_state:
            if verbose:
                print('Cache is not actual')

            if with_preprocessing:
                preprocessing_strategy = _preprocessing_strategy(preprocessed_data.data_type, self.model)().fit(
                    preprocessed_data.features)
                preprocessed_data.features = preprocessing_strategy.apply(preprocessed_data.features)
            else:
                preprocessing_strategy = None

            cached_model, model_predict = self.model.fit(data=preprocessed_data)
            self.cache.append(CachedState(preprocessor=copy(preprocessing_strategy),
                                          model=cached_model))
        else:
            if verbose:
                print('Model were obtained from cache')

            if with_preprocessing:
                preprocessing_strategy = self.cache.actual_cached_state.preprocessor
                preprocessed_data.features = preprocessing_strategy.apply(preprocessed_data.features)

            model_predict = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                               data=preprocessed_data)
        return model_predict

    @property
    def ordered_subnodes_hierarchy(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes

    def fine_tune(self, input_data: InputData, iterations: int = 30):
        raise NotImplementedError()


class FittedModelCache:
    def __init__(self, related_node: Node):
        self._local_cached_models = {}
        self._related_node_ref = related_node

    def append(self, fitted_model):
        self._local_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    def import_from_other_cache(self, other_cache: 'FittedModelCache'):
        for entry_key in other_cache._local_cached_models.keys():
            self._local_cached_models[entry_key] = other_cache._local_cached_models[entry_key]

    def clear(self):
        self._local_cached_models = {}

    @property
    def actual_cached_state(self):
        found_model = self._local_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


class SharedCache(FittedModelCache):
    def __init__(self, related_node: Node, global_cached_models: dict):
        super().__init__(related_node)
        self._global_cached_models = global_cached_models

    def append(self, fitted_model):
        super().append(fitted_model)
        if self._global_cached_models is not None:
            self._global_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    @property
    def actual_cached_state(self):
        found_model = super().actual_cached_state

        if not found_model and self._global_cached_models:
            found_model = self._global_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


# TODO: discuss about the usage of NodeGenerator
class NodeGenerator:
    @staticmethod
    def primary_node(model_type: str) -> Node:
        return PrimaryNode(model_type=model_type)

    @staticmethod
    def secondary_node(model_type: str,
                       nodes_from: Optional[List[Node]] = None) -> Node:
        return SecondaryNode(nodes_from=nodes_from, model_type=model_type)


def _preprocessing_strategy(dataset_type: DataTypesEnum, model: Model):
    _preprocessing_for_tasks = {
        DataTypesEnum.ts: DefaultStrategy,
        DataTypesEnum.table: Scaling,
        DataTypesEnum.ts_lagged_table: Scaling,
        DataTypesEnum.ts_lagged_3d: LaggedTimeSeriesFeature3dStrategy,
    }
    if 'without_preprocessing' not in model.metadata.tags:
        return _preprocessing_for_tasks[dataset_type]
    else:
        return DefaultStrategy


class PrimaryNode(Node):
    def __init__(self, model_type: str):
        model = Model(model_type=model_type)
        super().__init__(nodes_from=None, model=model)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        if verbose:
            print(f'Trying to fit primary node with model: {self.model}')

        model_predict = self._fit_using_cache(input_data=input_data, verbose=verbose)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict, task=input_data.task,
                          data_type=self.model.output_datatype(input_data.data_type))

    def predict(self, input_data: InputData, verbose=False) -> OutputData:
        if verbose:
            print(f'Predict in primary node by model: {self.model}')
        if not self.cache:
            raise ValueError('Model must be fitted before predict')

        preprocessed_data = transformation_function_for_data(input_data.data_type,
                                                             self.model.metadata.input_types)(input_data)

        preprocessed_data.features = self.cache.actual_cached_state.preprocessor.apply(preprocessed_data.features)

        predict_train = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                           data=preprocessed_data)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=predict_train, task=input_data.task,
                          data_type=self.model.output_datatype(input_data.data_type))

    def fine_tune(self, input_data: InputData, iterations: int = 30):
        preprocessed_data = transformation_function_for_data(input_data.data_type,
                                                             self.model.metadata.input_types)(input_data)

        preprocessing_strategy = _preprocessing_strategy(preprocessed_data.data_type, self.model)().fit(
            preprocessed_data.features)
        preprocessed_data.features = preprocessing_strategy.apply(preprocessed_data.features)

        fitted_model, predict_train = self.model.fine_tune(preprocessed_data, iterations=iterations)
        self.cache.append(CachedState(preprocessor=copy(preprocessing_strategy),
                                      model=fitted_model))


class SecondaryNode(Node):
    def __init__(self, nodes_from: Optional[List['Node']],
                 model_type: str):
        model = Model(model_type=model_type)
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, model=model)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError()
        parent_results = []

        if verbose:
            print(f'Fit all parent nodes in secondary node with model: {self.model}')

        target = input_data.target
        parent_nodes = self._nodes_from_with_fixed_order()

        if any(['affects_target' in parent_node.model.metadata.tags for parent_node in parent_nodes]):
            if len(parent_nodes) == 1:
                # is the previous model is the model that changes target
                parent_result = parent_nodes[0].fit(input_data=input_data)
                target = parent_result.predict

                parent_results.append(parent_result)
            else:
                raise NotImplementedError()

        else:
            for parent in parent_nodes:
                parent_results.append(parent.fit(input_data=input_data))

        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        if verbose:
            print(f'Trying to fit secondary node with model: {self.model}')

        model_predict = self._fit_using_cache(input_data=secondary_input, with_preprocessing=False)

        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=model_predict,
                          task=input_data.task,
                          data_type=self.model.output_datatype(input_data.data_type))

    def predict(self, input_data: InputData, verbose=False) -> OutputData:
        if len(self.nodes_from) == 0:
            raise ValueError('')
        parent_results = []
        if verbose:
            print(f'Obtain predictions from all parent nodes: {self.model}')
        parent_nodes = self._nodes_from_with_fixed_order()
        target = input_data.target

        #TODO refactor
        if any(['affects_target' in parent_node.model.metadata.tags for parent_node in parent_nodes]):
            if len(parent_nodes) == 1:
                # is the previous model is the model that changes target
                parent_result = parent_nodes[0].predict(input_data=input_data)
                target = parent_result.predict

                parent_results.append(parent_result)
            else:
                raise NotImplementedError()

        else:
            for parent in parent_nodes:
                parent_results.append(parent.predict(input_data=input_data))

        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)
        if verbose:
            print(f'Obtain prediction in secondary node with model: {self.model}')

        preprocessed_data = transformation_function_for_data(input_data.data_type,
                                                             self.model.metadata.input_types)(secondary_input)

        evaluation_result = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                               data=preprocessed_data)
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=evaluation_result,
                          data_type=self.model.output_datatype(input_data.data_type),
                          task=input_data.task)

    def fine_tune(self, input_data: InputData, iterations: int = 30):
        parent_results = []
        for parent in self._nodes_from_with_fixed_order():
            parent_results.append(parent.predict(input_data=input_data))

        target = input_data.target
        secondary_input = Data.from_predictions(outputs=parent_results,
                                                target=target)

        preprocessed_data = transformation_function_for_data(input_data.data_type,
                                                             self.model.metadata.input_types)(secondary_input)

        fitted_model, predict_train = self.model.fine_tune(preprocessed_data, iterations=iterations)
        self.cache.append(CachedState(preprocessor=None,
                                      model=fitted_model))
