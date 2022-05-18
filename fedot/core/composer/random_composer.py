from copy import copy
from random import randint
from typing import (Any, Callable, List, Optional)

from numpy import random

from fedot.core.composer.composer import ComposerRequirements, Composer
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.optimisers.objective import Objective, ObjectiveEvaluate
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Node, Pipeline


class RandomSearchComposer(Composer):
    def __init__(self, optimiser: 'RandomSearchOptimiser',
                 composer_requirements: ComposerRequirements,
                 metrics: Optional[Callable] = None):
        super().__init__(optimiser, composer_requirements)
        self._metrics = metrics

    def compose_pipeline(self, data: InputData) -> Pipeline:
        train_data = data
        test_data = data

        def metric(pipeline: Pipeline) -> float:
            pipeline.fit(train_data)
            return self._metrics(pipeline, test_data)

        objective = Objective(metric)
        objective_eval = ObjectiveEvaluate(objective)

        best_pipeline = self.optimiser.optimise(objective_eval)
        return best_pipeline


def nodes_to_pipeline(nodes: List[Node]) -> Pipeline:
    pipeline = Pipeline()
    [pipeline.add_node(nodes) for nodes in nodes]
    return pipeline


class RandomGraphFactory:
    def __init__(self,
                 primary_candidates: List[Any], secondary_candidates: List[Any],
                 primary_node_func: Callable = PrimaryNode, secondary_node_func: Callable = SecondaryNode):
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func
        self.__primary_candidates = primary_candidates
        self.__secondary_candidates = secondary_candidates

    def __call__(self):
        return self.random_nodeset()

    def random_nodeset(self) -> Pipeline:
        new_set = []

        # random primary nodes
        num_of_primary = randint(1, len(self.__primary_candidates))
        for _ in range(num_of_primary):
            new_set.append(self.random_primary())

        # random final node
        if len(new_set) > 1:
            parent_nodes = copy(new_set)
            final_node = self.random_secondary(parent_nodes)
            new_set.append(final_node)

        return nodes_to_pipeline(new_set)

    def random_primary(self):
        return self.__primary_node_func(random.choice(self.__primary_candidates))

    def random_secondary(self, parent_nodes):
        return self.__secondary_node_func(random.choice(self.__secondary_candidates), parent_nodes)


class RandomSearchOptimiser:

    def __init__(self, iter_num: int, random_pipeline_factory: Callable[..., Pipeline]):
        self._factory = random_pipeline_factory
        self._iter_num = iter_num
        self._log = default_log(__name__)

    def optimise(self, objective_evaluator: ObjectiveEvaluate) -> Pipeline:
        best_metric_value = 1000
        best_set = []
        history = []
        for i in range(self._iter_num):
            self._log.info(f'Iter {i}')
            new_pipeline = self._factory()
            new_metric_value = objective_evaluator(new_pipeline).value
            new_metric_value = round(new_metric_value, 3)

            history.append((new_pipeline, new_metric_value))

            self._log.info(f'Try {new_metric_value} with length {len(new_pipeline.nodes)}')
            if new_metric_value < best_metric_value:
                best_metric_value = new_metric_value
                best_set = new_pipeline
                self._log.info(f'Better pipeline found: metric {best_metric_value}')

        return best_set
