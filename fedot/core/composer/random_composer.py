from copy import copy
from random import randint
from typing import (Any, Callable, List, Sequence)

from numpy import random

from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.fitness import Fitness
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.objective import Objective, ObjectiveFunction
from fedot.core.optimisers.optimizer import GraphOptimizer
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Node, Pipeline


class RandomSearchComposer(Composer):
    def __init__(self, optimizer: 'RandomSearchOptimizer'):
        super().__init__(optimizer=optimizer)

    def compose_pipeline(self, data: InputData) -> Pipeline:
        train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

        def prepared_objective(pipeline: Pipeline) -> Fitness:
            pipeline.fit(train_data)
            return self.optimizer.objective(pipeline, reference_data=test_data)

        best_pipeline = self.optimizer.optimise(prepared_objective)[0]
        return best_pipeline


def nodes_to_pipeline(nodes: List[Node]) -> Pipeline:
    pipeline = Pipeline()
    [pipeline.add_node(nodes) for nodes in nodes]
    return pipeline


class RandomGraphFactory:
    def __init__(self,
                 primary_candidates: Sequence[Any], secondary_candidates: Sequence[Any],
                 primary_node_func: Callable = PrimaryNode, secondary_node_func: Callable = SecondaryNode):
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func
        self.__primary_candidates = list(primary_candidates)
        self.__secondary_candidates = list(secondary_candidates)

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


class RandomSearchOptimizer(GraphOptimizer):

    def __init__(self, objective: Objective,
                 requirements: PipelineComposerRequirements,
                 ):
        super().__init__(objective, requirements=requirements)
        self._factory = RandomGraphFactory(requirements.primary, requirements.secondary)

    def optimise(self, objective: ObjectiveFunction) -> Sequence[Pipeline]:
        best_metric_value = 1000
        best_set = None
        history = []
        for i in range(self.requirements.num_of_generations):
            new_pipeline = self._factory()
            new_metric_value = objective(new_pipeline).value
            new_metric_value = round(new_metric_value, 3)
            if new_metric_value < best_metric_value:
                best_metric_value = new_metric_value
                best_set = new_pipeline

            history.append((new_pipeline, new_metric_value))
            self.log.info(f'Iter {i}: best metric {best_metric_value},'
                          f'try {new_metric_value} with num nodes {new_pipeline.length}')
        return [best_set]
