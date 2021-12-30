import timeit
import warnings
from copy import deepcopy
from random import choice, randint
from typing import Any, Callable, List, Tuple, Union

from fedot.core.composer.constraint import constraint_function
from fedot.core.log import default_log
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.remote.remote_evaluator import RemoteEvaluator

MAX_ITERS = 1000


def random_graph(params, requirements, max_depth=None) -> OptGraph:
    max_depth = max_depth if max_depth else requirements.max_depth
    is_correct_graph = False
    graph = None
    n_iter = 0
    requirements = modify_requirements(requirements)

    while not is_correct_graph:
        graph = OptGraph()
        graph_root = OptNode(nodes_from=[],
                             content={'name': choice(requirements.secondary),
                                      'params': DEFAULT_PARAMS_STUB})
        graph.add_node(graph_root)
        graph_growth(graph, graph_root, requirements, max_depth)
        is_correct_graph = constraint_function(graph, params)
        n_iter += 1
        if n_iter > MAX_ITERS:
            warnings.warn(f'Random_graph generation failed for {n_iter} iterations.')
            raise ValueError(f'Random_graph generation failed. Cannot find valid graph'
                             f' with current params {params} and requirements {requirements}')

    return graph


def modify_requirements(requirements):
    """Function modify requirements if necessary.
    Example: Graph with only one primary node should consists of only one primary node
    without duplication, because this causes errors. Therefore minimum and maximum arity
    become equal to one.
    """
    if len(requirements.primary) == 1 and requirements.max_arity > 1:
        requirements.min_arity = requirements.max_arity = 1

    return requirements


def graph_growth(graph: OptGraph, node_parent: OptNode, requirements, max_depth: int):
    """Function create a graph and links between nodes"""
    offspring_size = randint(requirements.min_arity, requirements.max_arity)

    for offspring_node in range(offspring_size):
        height = graph.operator.distance_to_root_level(node_parent)
        is_max_depth_exceeded = height >= max_depth - 1
        is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
        if is_max_depth_exceeded or is_primary_node_selected:
            primary_node = OptNode(nodes_from=None,
                                   content={'name': choice(requirements.primary),
                                            'params': DEFAULT_PARAMS_STUB})
            node_parent.nodes_from.append(primary_node)
            graph.add_node(primary_node)
        else:
            secondary_node = OptNode(nodes_from=[],
                                     content={'name': choice(requirements.secondary),
                                              'params': DEFAULT_PARAMS_STUB})
            graph.add_node(secondary_node)
            node_parent.nodes_from.append(secondary_node)
            graph_growth(graph, secondary_node, requirements, max_depth)


def equivalent_subtree(graph_first: Any, graph_second: Any) -> List[Tuple[Any, Any]]:
    """Finds the similar subtree in two given trees"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and ((not node_first.nodes_from) or len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(graph_first.root_node, graph_second.root_node)
    assert isinstance(pairs_set, list)
    return pairs_set


def replace_subtrees(graph_first: Any, graph_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    node_from_graph_first_copy = deepcopy(node_from_first)

    summary_depth = layer_in_first + node_from_second.distance_to_primary_level
    if summary_depth <= max_depth and summary_depth != 0:
        graph_first.update_subtree(node_from_first, node_from_second)

    summary_depth = layer_in_second + node_from_first.distance_to_primary_level
    if summary_depth <= max_depth and summary_depth != 0:
        graph_second.update_subtree(node_from_second, node_from_graph_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def evaluate_individuals(individuals_set, objective_function, graph_generation_params,
                         is_multi_objective: bool, timer=None):
    logger = default_log('individuals evaluation logger')

    num_of_successful_evals = 0
    reversed_set = individuals_set[::-1]

    # TODO refactor
    fitter = RemoteEvaluator()
    pre_evaluated_objects = []
    if fitter.use_remote:
        logger.info('Remote fit used')
        restored_graphs = [graph_generation_params.adapter.restore(ind.graph) for ind in reversed_set]
        pre_evaluated_objects = fitter.compute_pipelines(restored_graphs)

    evaluated_individuals = []

    for ind_num, ind in enumerate(reversed_set):
        start_time = timeit.default_timer()

        graph = ind.graph
        if len(pre_evaluated_objects) > 0:
            graph = pre_evaluated_objects[ind_num]
        ind.fitness = calculate_objective(graph, objective_function,
                                          is_multi_objective, graph_generation_params)
        ind.computation_time = timeit.default_timer() - start_time
        if ind.fitness is not None:
            num_of_successful_evals += 1
            evaluated_individuals.append(ind)
        if timer is not None and num_of_successful_evals > 0:
            if timer.is_time_limit_reached():
                break
    if len(evaluated_individuals) == 0:
        raise AttributeError('Too much fitness evaluation errors. Composing stopped.')
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


def filter_duplicates(archive, population) -> List[Any]:
    filtered_archive = []
    for ind in archive.items:
        has_duplicate_in_pop = False
        for pop_ind in population:
            if ind.fitness == pop_ind.fitness:
                has_duplicate_in_pop = True
                break
        if not has_duplicate_in_pop:
            filtered_archive.append(ind)
    return filtered_archive


def duplicates_filtration(archive, population):
    return list(filter(lambda x: not any([x.fitness == pop_ind.fitness for pop_ind in population]), archive.items))


def clean_operators_history(population):
    for ind in population:
        ind.parent_operators = []
