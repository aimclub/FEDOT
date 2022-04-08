import gc
import multiprocessing
import timeit
from contextlib import closing
from types import SimpleNamespace
from typing import Union, Any, Dict, List

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.intermediate_metric import collect_intermediate_metric_for_nodes
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.pipelines.pipeline import Pipeline


def single_evaluating(reversed_individuals: List):
    """
    Evaluate individuals list in a single process

    :param reversed_individuals: list of individuals_context objects
    """
    evaluated_individuals = []
    num_of_successful_evals = 0
    for ind in reversed_individuals:
        individual_context = SimpleNamespace(**ind)
        start_time = timeit.default_timer()

        if len(individual_context.pre_evaluated_objects) > 0:
            individual_context.ind.graph = individual_context.pre_evaluated_objects[individual_context.ind_num]
        calculate_objective(individual_context)
        individual_context.metadata = {'computation_time': timeit.default_timer() - start_time}
        if individual_context.ind.fitness is not None:
            evaluated_individuals.append(individual_context.ind)
            num_of_successful_evals += 1
        if individual_context.timer is not None and num_of_successful_evals > 0:
            if individual_context.timer.is_time_limit_reached():
                return evaluated_individuals
    return evaluated_individuals


def calculate_objective(individual_context: SimpleNamespace) -> Any:
    """
    Calculate objective function for a graph

    :param individual_context: context object for individual
    """
    if isinstance(individual_context.ind.graph, OptGraph):
        converted_object = individual_context.graph_generation_params.adapter.restore(individual_context.ind.graph)
    else:
        converted_object = individual_context.ind.graph
    calculated_fitness = individual_context.objective_function(converted_object)

    if calculated_fitness is None:
        individual_context.ind.fitness = None
        return
    else:
        if individual_context.is_multi_objective:
            fitness = MultiObjFitness(values=calculated_fitness,
                                      weights=tuple([-1 for _ in range(len(calculated_fitness))]))
        else:
            fitness = calculated_fitness[0]
        individual_context.ind.fitness = fitness

    if individual_context.collect_intermediate_metric:
        collect_intermediate_metric_for_nodes(converted_object,
                                              individual_context.objective_function.keywords['reference_data'],
                                              individual_context.objective_function.keywords['cv_folds'],
                                              individual_context.objective_function.keywords['metrics'][0],
                                              individual_context.objective_function.keywords.get('validation_blocks',
                                                                                                 None))
        individual_context.ind.graph = individual_context.graph_generation_params.adapter.adapt(converted_object)

    if isinstance(converted_object, Pipeline):
        # enforce memory cleaning
        converted_object.unfit()
    gc.collect()


def determine_n_jobs(n_jobs, logger) -> int:
    """
    Cut n_jobs parameter if it bigger than the max CPU count

    :param n_jobs: num of process
    :param logger: log object
    """
    if n_jobs > multiprocessing.cpu_count() or n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs


def multiprocessing_mapping(n_jobs, reversed_individuals):
    """
    Evaluate individuals list in multiprocessing mode

    :param n_jobs: num of process
    :param reversed_individuals: list of individuals_context objects
    """
    with closing(multiprocessing.Pool(n_jobs)) as pool:
        return list(pool.imap_unordered(individual_evaluation, reversed_individuals))


def individual_evaluation(individual_context: Dict) -> Union[Individual, None]:
    """
    Calculate objective function for a graph for multiprocessing

    :param individual_context: context object for individual
    """
    start_time = timeit.default_timer()
    individual_context = SimpleNamespace(**individual_context)
    if individual_context.timer is not None and individual_context.timer.is_time_limit_reached():
        return
    if len(individual_context.pre_evaluated_objects) > 0:
        individual_context.ind.graph = individual_context.pre_evaluated_objects[individual_context.ind_num]
    replace_n_jobs_in_nodes(individual_context.ind.graph)
    calculate_objective(individual_context)
    individual_context.ind.metadata['computation_time'] = timeit.default_timer() - start_time
    return individual_context.ind


def replace_n_jobs_in_nodes(graph: OptGraph):
    """ Function to prevent memory overflow due to many processes running in time """
    for node in graph.nodes:
        if 'n_jobs' in node.content['params']:
            node.content['params']['n_jobs'] = 1
        if 'num_threads' in node.content['params']:
            node.content['params']['num_threads'] = 1
