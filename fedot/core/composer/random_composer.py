from copy import copy
from functools import partial
from random import randint
from typing import (Any, Callable, List, Optional)

from numpy import random

from fedot.core.composer.composer import ComposerRequirements, Composer
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Node, Pipeline


class RandomSearchComposer(Composer):
    def __init__(self,
                 composer_requirements: ComposerRequirements,
                 iter_num: int = 10,
                 metrics: Optional[Callable] = None):
        self._metrics = metrics
        self.optimiser = RandomSearchOptimiser(iter_num,
                                               PrimaryNode, SecondaryNode,
                                               composer_requirements.primary, composer_requirements.secondary)
        super().__init__(self.optimiser, composer_requirements)

    def compose_pipeline(self, data: InputData) -> Pipeline:
        train_data = data
        test_data = data
        metric_function_for_nodes = partial(metric_for_nodes,
                                            metric_function=self._metrics,
                                            train_data=train_data,
                                            test_data=test_data)

        best_nodes_set = self.optimiser.optimise(metric_function_for_nodes)

        best_pipeline = Pipeline()
        [best_pipeline.add_node(nodes) for nodes in best_nodes_set]

        return best_pipeline


def nodes_to_pipeline(nodes: List[Node]) -> Pipeline:
    pipeline = Pipeline()
    [pipeline.add_node(nodes) for nodes in nodes]
    return pipeline


def metric_for_nodes(metric_function, nodes: List[Node], train_data: InputData, test_data: InputData) -> float:
    pipeline = nodes_to_pipeline(nodes)
    pipeline.fit(input_data=train_data)
    return metric_function(pipeline, test_data)


class RandomSearchOptimiser:
    def __init__(self, iter_num: int,
                 primary_node_func: Callable, secondary_node_func: Callable,
                 primary_candidates: List[Any], secondary_candidates: List[Any]):
        self.__iter_num = iter_num
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func
        self.__primary_candidates = primary_candidates
        self.__secondary_candidates = secondary_candidates

    def optimise(self, metric_function_for_nodes) -> List[Node]:
        best_metric_value = 1000
        best_set = []
        history = []
        for i in range(self.__iter_num):
            print(f'Iter {i}')
            new_nodeset = self._random_nodeset()
            new_metric_value = round(metric_function_for_nodes(nodes=new_nodeset), 3)
            history.append((new_nodeset, new_metric_value))

            print(f'Try {new_metric_value} with length {len(new_nodeset)}')
            if new_metric_value < best_metric_value:
                best_metric_value = new_metric_value
                best_set = new_nodeset
                print(f'Better pipeline found: metric {best_metric_value}')

        return best_set

    def _random_nodeset(self) -> List[Node]:
        new_set = []

        # random primary nodes
        num_of_primary = randint(1, len(self.__primary_candidates))
        random_first_operations = random.choice(self.__primary_candidates, num_of_primary, replace=False)
        [new_set.append(self.__primary_node_func(operation)) for operation in random_first_operations]

        # random final node
        if len(new_set) > 1:
            random_final_operation = random.choice(self.__secondary_candidates, replace=False)
            parent_nodes = copy(new_set)
            new_set.append(self.__secondary_node_func(random_final_operation, parent_nodes))

        return new_set
