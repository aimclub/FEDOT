from copy import copy
from functools import partial
from random import randint
from typing import (
    List,
    Callable,
    Optional,
    Any
)

from numpy import random

from core.composer.chain import Chain, Node
from core.composer.composer import ComposerRequirements
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.data import train_test_data_setup
from experiments.chain_template import chain_template_random, fit_template, real_chain


class History:
    def __init__(self):
        self.values = []

    def on_new_value(self, value):
        self.values.append(value)

    def clear(self):
        self.values = []


class RandomSearchComposer:
    def __init__(self, iter_num: int = 10):
        self.__iter_num = iter_num

    def compose_chain(self, data: InputData,
                      initial_chain: Optional[Chain],
                      composer_requirements: ComposerRequirements,
                      metrics: Optional[Callable],
                      history_callback: History) -> Chain:
        train_data, test_data = train_test_data_setup(data, 0.8)
        metric_function_for_nodes = partial(metric_for_nodes,
                                            metric_function=metrics,
                                            train_data=train_data,
                                            test_data=test_data)
        optimiser = RandomSearchOptimiser(self.__iter_num,
                                          NodeGenerator.primary_node,
                                          NodeGenerator.secondary_node)
        best_nodes_set, history = optimiser.optimise(metric_function_for_nodes,
                                                     composer_requirements.primary,
                                                     composer_requirements.secondary,
                                                     initial_solution=initial_chain,
                                                     history_callback=history_callback)

        best_chain = Chain()
        [best_chain.add_node(nodes) for nodes in best_nodes_set]

        return best_chain


def nodes_to_chain(nodes: List[Node]) -> Chain:
    chain = Chain()
    [chain.add_node(nodes) for nodes in nodes]
    return chain


def metric_for_nodes(metric_function, nodes: List[Node], train_data: InputData, test_data: InputData) -> float:
    chain = nodes_to_chain(nodes)
    chain.fit(input_data=train_data)
    return metric_function(chain, test_data)


class RandomSearchOptimiser:
    def __init__(self, iter_num: int, primary_node_func: Callable,
                 secondary_node_func: Callable):
        self.__iter_num = iter_num
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func

    def optimise(self, metric_function_for_nodes,
                 primary_candidates: List[Any],
                 secondary_candidates: List[Any],
                 initial_solution: Optional[Chain],
                 history_callback: History):

        if initial_solution is not None:
            best_set = initial_solution.nodes
            best_metric_value = metric_function_for_nodes(nodes=best_set)
        else:
            best_metric_value = 1000
            best_set = []
        history = []
        for i in range(self.__iter_num):
            print(f'Iter {i}')
            # new_nodeset = self._random_nodeset(primary_candidates, secondary_candidates)
            depth = random.choice([_ for _ in range(2, 5)])
            models_per_level = 5
            new_nodeset = _random_chain(primary_candidates, depth=depth,
                                        models_per_level=models_per_level)
            new_metric_value = round(metric_function_for_nodes(nodes=new_nodeset), 3)
            history.append((new_nodeset, new_metric_value))

            print(f'Try {new_metric_value} with length {len(new_nodeset)}')
            if new_metric_value < best_metric_value:
                best_metric_value = new_metric_value
                best_set = new_nodeset
                print(f'Better chain found: metric {best_metric_value}')
            history_callback.on_new_value(abs(best_metric_value))
        return best_set, history

    def _random_nodeset(self, primary_requirements: List[Any], secondary_requirements: List[Any]) -> List[Node]:
        new_set = []

        # random primary nodes
        num_of_primary = randint(1, len(primary_requirements))
        random_first_models = random.choice(primary_requirements, num_of_primary, replace=False)
        [new_set.append(self.__primary_node_func(model)) for model in random_first_models]

        # random final node
        if len(new_set) > 1:
            random_final_model = random.choice(secondary_requirements, replace=False)
            parent_nodes = copy(new_set)
            new_set.append(self.__secondary_node_func(random_final_model, parent_nodes))

        return new_set


def _random_chain(requirements, depth=4, models_per_level=4):
    template = chain_template_random(model_types=requirements,
                                     depth=depth, models_per_level=models_per_level,
                                     samples=1000, features=10)
    fit_template(template, classes=2, skip_fit=True)
    final_chain = real_chain(template)

    return final_chain.nodes
