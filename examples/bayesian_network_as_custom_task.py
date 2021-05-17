import datetime
import os
import random
from functools import partial

import numpy as np
import pandas as pd

from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.chain_validation import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import (
    GPChainOptimiser,
    GPChainOptimiserParameters,
    GeneticSchemeTypesEnum)
from fedot.core.composer.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.composer.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.graphs.graph import GraphObject
from fedot.core.graphs.graph_node import PrimaryGraphNode, SecondaryGraphNode
from fedot.core.log import default_log
from fedot.core.utils import fedot_project_root

random.seed(1)
np.random.seed(1)


def custom_metric(network: GraphObject, data: pd.DataFrame):
    network.show()
    nodes = data.columns.to_list()
    _, labels = chain_as_nx_graph(network)
    existing_variables_num = -len([label for label in list(labels.values()) if str(label) in nodes])
    return [existing_variables_num]


def _has_no_duplicates(graph):
    _, labels = chain_as_nx_graph(graph)
    list_of_nodes = [str(node) for node in labels.values()]
    if len(list_of_nodes) != len(set(list_of_nodes)):
        raise ValueError('Chain has duplicates')
    return True


def custom_mutation(chain: GraphObject, requirements: GPComposerRequirements,
                    chain_generation_params: ChainGenerationParams, max_depth: int = None):
    for _ in range(random.randint(1, 5)):
        rid = random.choice(range(len(chain.nodes)))
        random_node = chain.nodes[rid]
        other_random_node = chain.nodes[random.choice(range(len(chain.nodes)))]
        if random_node.operation != other_random_node.operation:
            chain.operator.connect_nodes(random_node, other_random_node)
    return chain


def run_bayesian(max_lead_time: datetime.timedelta = datetime.timedelta(minutes=0.2)):
    data = pd.read_csv(os.path.join(fedot_project_root(), 'examples', 'data', 'geo_encoded.csv'))
    nodes_types = ['Tectonic regime', 'Period', 'Lithology',
                   'Structural setting', 'Gross', 'Netpay',
                   'Porosity', 'Permeability', 'Depth']
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    initial = GraphObject(nodes=[PrimaryGraphNode(_) for _ in nodes_types])

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=10,
        max_depth=10, pop_size=20, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[custom_mutation],
        crossover_types=[CrossoverTypesEnum.none],
        regularization_type=RegularizationTypesEnum.none)

    chain_generation_params = ChainGenerationParams(
        chain_class=GraphObject,
        primary_node_func=PrimaryGraphNode,
        secondary_node_func=SecondaryGraphNode,
        rules_for_constraint=rules)

    optimizer = GPChainOptimiser(
        chain_generation_params=chain_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_chain=initial,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_network = optimizer.optimise(partial(custom_metric, data=data))

    optimized_network.show()


if __name__ == '__main__':
    run_bayesian()
