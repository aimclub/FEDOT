from copy import deepcopy
from typing import (Any, Callable, List, Optional)

from fedot.core.composer.constraint import constraint_function
from fedot.core.utils import ComparableEnum as Enum


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


def regularized_population(reg_type: RegularizationTypesEnum, population: List[Any],
                           objective_function: Callable,
                           chain_class: Any, size: Optional[int] = None) -> List[Any]:
    if reg_type == RegularizationTypesEnum.decremental:
        additional_inds = decremental_regularization(population, objective_function, chain_class, size)
        return population + additional_inds
    elif reg_type == RegularizationTypesEnum.none:
        return population
    else:
        raise ValueError(f'Required regularization type not found: {type}')


def decremental_regularization(population: List[Any], objective_function: Callable,
                               chain_class: Any, size: Optional[int] = None) -> List[Any]:
    size = size if size else len(population)
    additional_inds = []
    prev_nodes_ids = []
    for ind in population:
        ind_subtrees = [node for node in ind.nodes if node != ind.root_node]
        subtrees = [chain_class(deepcopy(node.ordered_subnodes_hierarchy)) for node in ind_subtrees if
                    is_fitted_subtree(node, prev_nodes_ids)]
        additional_inds += subtrees
        prev_nodes_ids += [subtree.root_node.descriptive_id for subtree in subtrees]

    additional_inds = [ind for ind in additional_inds if constraint_function(ind)]

    for additional_ind in additional_inds:
        additional_ind.fitness = objective_function(additional_ind)

    if additional_inds and len(additional_inds) > size:
        additional_inds = sorted(additional_inds, key=lambda ind: ind.fitness)[:size]

    return additional_inds


def is_fitted_subtree(node: Any, prev_nodes_ids: List[Any]) -> bool:
    return node.nodes_from and not node.descriptive_id in prev_nodes_ids and node.cache.actual_cached_state
