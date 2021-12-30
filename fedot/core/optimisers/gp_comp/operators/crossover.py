from copy import deepcopy
from random import choice, random
from typing import TYPE_CHECKING, Any, Callable, List, Union

from fedot.core.composer.constraint import constraint_function
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree, replace_subtrees
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.utils import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams

MAX_NUM_OF_ATTEMPTS = 100


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


def will_crossover_be_applied(graph_first, graph_second, crossover_prob, crossover_type) -> bool:
    return not (graph_first is graph_second or
                random() > crossover_prob or
                crossover_type == CrossoverTypesEnum.none)


def crossover(types: List[Union[CrossoverTypesEnum, Callable]],
              ind_first: Individual, ind_second: Individual,
              max_depth: int, log: Log,
              crossover_prob: float = 0.8, params: 'GraphGenerationParams' = None) -> Any:
    crossover_type = choice(types)
    is_custom_crossover = isinstance(crossover_type, Callable)
    try:
        if will_crossover_be_applied(ind_first.graph, ind_second.graph, crossover_prob, crossover_type):
            if crossover_type in crossover_by_type.keys() or is_custom_crossover:
                for _ in range(MAX_NUM_OF_ATTEMPTS):
                    if is_custom_crossover:
                        crossover_func = crossover_type
                    else:
                        crossover_func = crossover_by_type[crossover_type]
                    new_inds = []

                    is_custom_operator = isinstance(ind_first, OptGraph)
                    input_obj_first = deepcopy(ind_first.graph)
                    input_obj_second = deepcopy(ind_first.graph)
                    if is_custom_operator:
                        input_obj_first = params.adapter.restore(input_obj_first)
                        input_obj_second = params.adapter.restore(input_obj_second)

                    new_graphs = crossover_func(input_obj_first,
                                                input_obj_second, max_depth)

                    if is_custom_operator:
                        for graph_id, graph in enumerate(new_graphs):
                            new_graphs[graph_id] = params.adapter.adapt(graph)

                    are_correct = \
                        all([constraint_function(new_graph, params)
                             for new_graph in new_graphs])

                    if are_correct:
                        operator = ParentOperator(operator_type='crossover',
                                                  operator_name=str(crossover_type),
                                                  parent_objects=[ind_first, ind_second])
                        for graph in new_graphs:
                            new_ind = Individual(graph)
                            new_ind.parent_operators = []
                            new_ind.parent_operators.extend(ind_first.parent_operators)
                            new_ind.parent_operators.extend(ind_second.parent_operators)
                            new_ind.parent_operators.append(operator)
                            new_inds.append(new_ind)
                        return new_inds
            else:
                raise ValueError(f'Required crossover type not found: {crossover_type}')

            log.debug('Number of crossover attempts exceeded. '
                      'Please check composer requirements for correctness.')
    except Exception as ex:
        log.error(f'Crossover ex: {ex}')

    graph_first_copy = deepcopy(ind_first)
    graph_second_copy = deepcopy(ind_second)
    return graph_first_copy, graph_second_copy


def subtree_crossover(graph_first: Any, graph_second: Any, max_depth: int) -> Any:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    random_layer_in_graph_first = choice(range(graph_first.depth))
    min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_second.depth > 1 else 0
    random_layer_in_graph_second = choice(range(min_second_layer, graph_second.depth))

    node_from_graph_first = choice(graph_first.operator.nodes_from_layer(random_layer_in_graph_first))
    node_from_graph_second = choice(graph_second.operator.nodes_from_layer(random_layer_in_graph_second))

    replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                     random_layer_in_graph_first, random_layer_in_graph_second, max_depth)

    return graph_first, graph_second


def one_point_crossover(graph_first: Any, graph_second: Any, max_depth: int) -> Any:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
    if pairs_of_nodes:
        node_from_graph_first, node_from_graph_second = choice(pairs_of_nodes)

        layer_in_graph_first = \
            graph_first.root_node.distance_to_primary_level - node_from_graph_first.distance_to_primary_level
        layer_in_graph_second = \
            graph_second.root_node.distance_to_primary_level - node_from_graph_second.distance_to_primary_level

        replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                         layer_in_graph_first, layer_in_graph_second, max_depth)
    return graph_first, graph_second


crossover_by_type = {
    CrossoverTypesEnum.subtree: subtree_crossover,
    CrossoverTypesEnum.one_point: one_point_crossover,
}
