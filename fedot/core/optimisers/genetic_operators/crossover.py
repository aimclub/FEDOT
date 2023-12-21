from copy import deepcopy
from functools import WRAPPER_ASSIGNMENTS
from random import choice

from typing import Callable, Tuple

from fedot.core.optimisers.genetic_operators.atomized_operators_wrapper import \
    extract_graphs_from_atomized, insert_graphs_to_atomized
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from golem.core.adapter import register_native
from golem.core.optimisers.genetic.operators.base_mutations import \
    single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from golem.core.optimisers.genetic.operators.crossover import CrossoverCallable, one_point_crossover, subtree_crossover
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


def atomized_crossover(crossover_fun: CrossoverCallable) -> CrossoverCallable:
    def crossover_for_atomized_graph(graph1: OptGraph,
                                     graph2: OptGraph,
                                     max_depth: int
                                     ) -> Tuple[OptGraph, OptGraph]:
        graphs = [deepcopy(graph1), deepcopy(graph2)]
        graphs1, graphs2 = map(extract_graphs_from_atomized, graphs)
        node_uid1, graph_for_crossover1 = choice(list(graphs1.items()))
        node_uid2, graph_for_crossover2 = choice(list(graphs2.items()))

        graph_after_crossover1, graph_after_crossover2 = crossover_fun(graph_for_crossover1, graph_for_crossover2, max_depth)

        new_graph1 = insert_graphs_to_atomized(graph1, node_uid1, graph_after_crossover1)
        new_graph2 = insert_graphs_to_atomized(graph2, node_uid2, graph_after_crossover2)
        return new_graph1, new_graph2

    # TODO use functools.wraps. now it brokes something in GOLEM.
    for attr in WRAPPER_ASSIGNMENTS:
        setattr(crossover_for_atomized_graph, attr, getattr(crossover_fun, attr))
    crossover_for_atomized_graph.__wrapped__ = crossover_fun

    return crossover_for_atomized_graph


fedot_one_point_crossover = register_native(atomized_crossover(one_point_crossover))
fedot_subtree_crossover = register_native(atomized_crossover(subtree_crossover))
