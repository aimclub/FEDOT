from datetime import timedelta, datetime
from functools import partial
from itertools import product
from typing import Callable, Sequence, Optional

import networkx as nx

from examples.advanced.abstract_graph_evolution.graph_metrics import get_edit_dist_metric, spectral_dist, num_nodes_diff
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
    return [
        partial(nx.gnp_random_graph, p=0.15),
        nx.gn_graph,
    ]


def run_experiments(graph_generators: Sequence[DiGraphGenerator],
                    graph_sizes: Sequence[int] = (10, 30, 100,),
                    num_node_kinds: int = 10,
                    num_trials: int = 10,
                    trial_timeout: Optional[int] = None,
                    visualize: bool = False,
                    ):
    for graph_generator, num_nodes in product(graph_generators, graph_sizes):
        for i in range(num_trials):
            start_time = datetime.now()
            print(f'Trial #{i} with graph_size={num_nodes} for graph generator '
                  f'{getattr(graph_generator, "__name__", str(graph_generator))}')

            target_graph = graph_generator(num_nodes)
            found_graph, history = run_experiment(target_graph, num_nodes,
                                                  timeout=timedelta(minutes=trial_timeout))

            duration = datetime.now() - start_time
            print(f'Trial #{i} finished, spent time: {duration}')
            if visualize:
                history.show.fitness_line_interactive()
                pass


def run_experiment(target_graph: nx.DiGraph,
                   num_nodes: int,
                   timeout: Optional[timedelta] = None):
    num_node_kinds: int = 3
    nodes_types = [f'V{i}' for i in range(1, num_node_kinds+1)]
    # TODO: simple initial pop
    initial = [OptGraph(OptNode(node_type)) for node_type in nodes_types]
    # initial = [BaseNetworkxAdapter().restore(ind) for ind in initial]
    # initial = [nx.star_graph(num_nodes)] * 5

    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types,
        max_arity=num_nodes,
        max_depth=num_nodes,

        keep_n_best=3,
        early_stopping_generations=20,
        timeout=timeout,
        max_pipeline_fit_time=timedelta(seconds=30),
        n_jobs=8,
    )

    optimiser_parameters = GPGraphOptimizerParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
    )

    graph_generation_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes, has_no_cycle,],
        available_node_types=nodes_types,
    )

    objective = Objective(
        quality_metrics={
            # 'edit_distance': get_edit_dist_metric(target_graph, requirements),
            # 'matrix_edit_dist': partial(matrix_edit_dist, target_graph),
            'spectral_adjacency': partial(spectral_dist, target_graph, kind='adjacency'),
            'spectral_laplacian': partial(spectral_dist, target_graph, kind='laplacian'),
            'spectral_laplacian_norm': partial(spectral_dist, target_graph, kind='laplacian_norm'),
        },
        complexity_metrics={
            'num_nodes_diff': partial(num_nodes_diff, target_graph),
        },
        is_multi_objective=True
    )

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
