import datetime
import os
import random
from functools import partial

import numpy as np
import pandas as pd

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utils import fedot_project_root

random.seed(1)
np.random.seed(1)


class CustomGraphModel(OptGraph):
    def evaluate(self, data: pd.DataFrame):
        nodes = data.columns.to_list()
        _, labels = graph_structure_as_nx_graph(self)
        return len(nodes)


class CustomGraphNode(OptNode):
    def __str__(self):
        return f'Node_{self.content["name"]}'


def custom_metric(graph: CustomGraphModel, data: pd.DataFrame):
    graph.show()
    existing_variables_num = -graph.depth - graph.evaluate(data)

    return [existing_variables_num]


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def custom_mutation(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            if random_node.nodes_from is not None and len(random_node.nodes_from) == 0:
                random_node.nodes_from = None
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph


def run_custom_example(timeout: datetime.timedelta = None):
    if not timeout:
        timeout = datetime.timedelta(minutes=1)
    data = pd.read_csv(os.path.join(fedot_project_root(), 'examples', 'data', 'custom_encoded.csv'))
    nodes_types = ['V1', 'V2', 'V3',
                   'V4', 'V5', 'V6',
                   'V7', 'V8', 'V9', 'V10']
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    initial = CustomGraphModel(nodes=[CustomGraphNode(nodes_from=None,
                                                      content={'name': node_type}) for node_type in nodes_types])

    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=10,
        max_depth=10, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.9, timeout=timeout)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[custom_mutation],
        crossover_types=[CrossoverTypesEnum.none],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_graph = optimiser.optimise(partial(custom_metric, data=data))
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    optimized_network.show()


if __name__ == '__main__':
    run_custom_example()
