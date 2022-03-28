import multiprocessing
import timeit
from contextlib import closing
from types import SimpleNamespace
from typing import Union, Any, Callable, Dict

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness


def single_evaluating(reversed_individuals):
    evaluated_individuals = []
    num_of_successful_evals = 0
    for ind in reversed_individuals:
        individual = SimpleNamespace(**ind)
        start_time = timeit.default_timer()

        graph = individual.ind.graph
        if len(individual.pre_evaluated_objects) > 0:
            graph = individual.pre_evaluated_objects[individual.ind_num]
        individual.ind.fitness = calculate_objective(graph, individual.objective_function,
                                                     individual.is_multi_objective, individual.graph_generation_params)
        individual.computation_time = timeit.default_timer() - start_time
        if individual.ind.fitness is not None:
            evaluated_individuals.append(individual.ind)
            num_of_successful_evals += 1
        if individual.timer is not None and num_of_successful_evals > 0:
            if individual.timer.is_time_limit_reached():
                return evaluated_individuals
    return evaluated_individuals


def calculate_objective(graph: Union[OptGraph, Any], objective_function: Callable,
                        is_multi_objective: bool,
                        graph_generation_params) -> Any:
    if isinstance(graph, OptGraph):
        converted_object = graph_generation_params.adapter.restore(graph)
    else:
        converted_object = graph
    calculated_fitness = objective_function(converted_object)
    if calculated_fitness is None:
        return None
    else:
        if is_multi_objective:
            fitness = MultiObjFitness(values=calculated_fitness,
                                      weights=tuple([-1 for _ in range(len(calculated_fitness))]))
        else:
            fitness = calculated_fitness[0]
    return fitness


def determine_n_jobs(n_jobs, logger):
    if n_jobs > multiprocessing.cpu_count() or n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs


def multiprocessing_mapping(n_jobs, reversed_set):
    with closing(multiprocessing.Pool(n_jobs)) as pool:
        return list(pool.imap_unordered(individual_evaluation, reversed_set))


def individual_evaluation(individual: Dict) -> Union[Individual, None]:
    start_time = timeit.default_timer()
    individual_ = SimpleNamespace(**individual)
    graph = individual_.ind.graph

    if individual_.timer is not None and individual_.timer.is_time_limit_reached():
        return

    if len(individual_.pre_evaluated_objects) > 0:
        graph = individual_.pre_evaluated_objects[individual_.ind_num]
    replace_n_jobs_in_nodes(graph)
    individual_.ind.fitness = calculate_objective(graph, individual_.objective_function,
                                                  individual_.is_multi_objective, individual_.graph_generation_params)
    individual_.ind.computation_time = timeit.default_timer() - start_time
    return individual_.ind


def replace_n_jobs_in_nodes(graph: OptGraph):
    """ Function to prevent memory overflow due to many processes running in time"""
    for node in graph.nodes:
        if 'n_jobs' in node.content['params']:
            node.content['params']['n_jobs'] = 1
        if 'num_threads' in node.content['params']:
            node.content['params']['num_threads'] = 1
