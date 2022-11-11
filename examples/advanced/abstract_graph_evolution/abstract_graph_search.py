from datetime import timedelta, datetime
from functools import partial
from itertools import product
from typing import Callable, Sequence, Optional, Dict

import networkx as nx
import numpy as np
from networkx import graph_edit_distance

from fedot.core.adapter.nx_adapter import BaseNetworkxAdapter
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams

NumNodes = int
DiGraphGenerator = Callable[[NumNodes], nx.DiGraph]


def graph_generators() -> Sequence[DiGraphGenerator]:
    return [nx.gn_graph,
            partial(nx.gnp_random_graph, p=0.02),
            partial(nx.gnp_random_graph, p=0.05),
            ]


def get_similarity_metric(target_graph: nx.DiGraph,
                          requirements: Optional[PipelineComposerRequirements] = None,
                          ) -> Callable[[nx.DiGraph], float]:

    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = requirements.max_pipeline_fit_time.seconds,
    else:
        upper_bound = None
        timeout = timedelta(seconds=60)

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.seconds,
                                 )
        return ged or upper_bound

    return metric


def run_experiments(graph_generators: Sequence[DiGraphGenerator],
                    graph_sizes: Sequence[int] = (10, 100, 1000),
                    num_node_kinds: int = 10,
                    num_trials: int = 10,
                    trial_timeout: Optional[int] = None,
                    visualize: bool = False,
                    ):
    for graph_generator, graph_size in product(graph_generators, graph_sizes):
        for i in range(num_trials):
            start_time = datetime.now()
            print(f'Trial #{i} with graph_size={graph_size} for graph generator '
                  f'{getattr(graph_generator, "__name__", str(graph_generator))}')

            target_graph = graph_generator(graph_size)
            found_graph, history = run_experiment(target_graph, timeout=timedelta(minutes=trial_timeout))

            duration = datetime.now() - start_time
            print(f'Trial #{i} finished, spent time: {duration}')
            if visualize:
                history.show.fitness_line_interactive()
                pass


def run_experiment(target_graph: nx.DiGraph,
                   timeout: Optional[timedelta] = None):
    num_node_kinds: int = 10
    nodes_types = [f'V{i}' for i in range(1, num_node_kinds+1)]
    # TODO: simple initial pop
    initial = [OptGraph(OptNode(node_type)) for node_type in nodes_types]
    initial = [BaseNetworkxAdapter().restore(ind) for ind in initial]

    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types,
        max_arity=100,
        max_depth=100,

        early_stopping_generations=20,
        timeout=timeout,
        max_pipeline_fit_time=timedelta(seconds=30),
        n_jobs=1,
    )

    optimiser_parameters = GPGraphOptimizerParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
    )

    graph_generation_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes, has_no_cycle,],
        available_node_types=nodes_types,
    )

    levenstein_metric = get_similarity_metric(requirements, target_graph)
    objective = Objective(quality_metrics={'edit_distance': levenstein_metric})

    optimiser = EvoGraphOptimizer(
        objective=objective,
        initial_graphs=initial,
        requirements=requirements,
        graph_optimizer_params=optimiser_parameters,
        graph_generation_params=graph_generation_params,
    )

    found_graphs = optimiser.optimise(objective)
    # found_networks = [graph_generation_params.adapter.restore(g) for g in found_graphs]

    return found_graphs[0], optimiser.history


if __name__ == '__main__':
    run_experiments(graph_generators(), trial_timeout=5, visualize=True)
