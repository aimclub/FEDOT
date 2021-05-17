import random

import numpy as np

from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.chain_validation import has_no_self_cycled_nodes
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import (
    GPChainOptimiser,
    GPChainOptimiserParameters,
    GeneticSchemeTypesEnum)
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.composer.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.graphs.graph import GraphObject
from fedot.core.graphs.graph_node import PrimaryGraphNode, SecondaryGraphNode

random.seed(1)
np.random.seed(1)


def custom_metric(network: GraphObject):
    _, labels = chain_as_nx_graph(network)

    return [-len(labels)]


def test_custom_graph_opt():
    nodes_types = ['A', 'B', 'C', 'D']
    rules = [has_no_self_cycled_nodes]

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.9)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth],
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
        requirements=requirements, initial_chain=None)

    optimized_network = optimizer.optimise(custom_metric)

    assert optimized_network is not None
    assert isinstance(optimized_network, GraphObject)
    assert isinstance(optimized_network.nodes[0], PrimaryGraphNode) or \
           isinstance(optimized_network.nodes[0], SecondaryGraphNode)

    assert 'A' in [str(_) for _ in optimized_network.nodes]
