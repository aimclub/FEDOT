import random
from itertools import repeat, cycle

import numpy as np
import pytest

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.validation_rules import has_no_self_cycled_nodes
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate

random.seed(1)
np.random.seed(1)


class CustomModel(Graph):
    def evaluate(self):
        return 0


class CustomNode(GraphNode):
    def __str__(self):
        return f'custom_{str(self.content["name"])}'


def custom_metric(custom_model: CustomModel):
    _, labels = graph_structure_as_nx_graph(custom_model)

    return -len(labels) + custom_model.evaluate()


def test_custom_graph_opt():
    """Test checks for the use case of custom graph optimisation:
    that it can be initialised without problem and returns sane result."""

    nodes_types = ['A', 'B', 'C', 'D']
    rules = [has_no_self_cycled_nodes]

    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.9)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(CustomModel, CustomNode),
        rules_for_constraint=rules)

    objective = Objective(custom_metric)
    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        objective=objective,
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=None)

    objective_eval = ObjectiveEvaluate(objective)
    optimized_graphs = optimiser.optimise(objective_eval)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graphs[0])

    assert optimized_network is not None
    assert isinstance(optimized_network, CustomModel)
    assert isinstance(optimized_network.nodes[0], CustomNode)
    assert len(optimized_network.nodes) > 1
    assert optimized_network.depth > 1
