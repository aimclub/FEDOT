from copy import deepcopy
from typing import (Any, Callable, List, Optional)

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_comp.gp_operators import evaluate_individuals
from fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.utils import ComparableEnum as Enum


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


def regularized_population(reg_type: RegularizationTypesEnum, population: List[Any],
                           objective_function: Callable,
                           chain_class: Any, size: Optional[int] = None, timer=None) -> List[Any]:
    if reg_type == RegularizationTypesEnum.decremental:
        additional_inds = decremental_regularization(population, objective_function, chain_class, size, timer=timer)
        return population + additional_inds
    elif reg_type == RegularizationTypesEnum.none:
        return population
    else:
        raise ValueError(f'Required regularization type not found: {type}')


def decremental_regularization(population: List[Any], objective_function: Callable,
                               chain_class: Any, size: Optional[int] = None, timer=None) -> List[Any]:
    size = size if size else len(population)
    additional_inds = []
    prev_nodes_ids = []
    for ind in population:
        ind_subtrees = [node for node in ind.nodes if node != ind.root_node]
        subtrees = [chain_class(deepcopy(node.ordered_subnodes_hierarchy())) for node in ind_subtrees if
                    is_fitted_subtree(node, prev_nodes_ids)]
        additional_inds += subtrees
        prev_nodes_ids += [subtree.root_node.descriptive_id for subtree in subtrees]

    additional_inds = [ind for ind in additional_inds if constraint_function(ind)]

    is_multi_obj = (population[0].fitness) is MultiObjFitness
    if additional_inds:
        evaluate_individuals(additional_inds, objective_function, is_multi_obj, timer=timer)

    if additional_inds and len(additional_inds) > size:
        additional_inds = sorted(additional_inds, key=lambda ind: ind.fitness)[:size]

    return additional_inds


def is_fitted_subtree(node: Any, prev_nodes_ids: List[Any]) -> bool:
    return node.nodes_from and node.descriptive_id not in prev_nodes_ids and node.fitted_model
