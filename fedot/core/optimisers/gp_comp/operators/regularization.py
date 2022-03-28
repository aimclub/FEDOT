from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from fedot.core.composer.constraint import constraint_function
from fedot.core.optimisers.gp_comp.gp_operators import evaluate_individuals
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


def regularized_population(reg_type: RegularizationTypesEnum, population: List[Any],
                           objective_function: Callable,
                           graph_generation_params: Any, size: Optional[int] = None, timer=None) -> List[Any]:
    if reg_type == RegularizationTypesEnum.decremental:
        additional_inds = decremental_regularization(population, objective_function,
                                                     graph_generation_params, size, timer=timer)
        return population + additional_inds
    elif reg_type == RegularizationTypesEnum.none:
        return population
    else:
        raise ValueError(f'Required regularization type not found: {type}')


def decremental_regularization(population: List[Individual], objective_function: Callable,
                               params: 'GraphGenerationParams',
                               size: Optional[int] = None, timer=None) -> List[Any]:
    size = size if size else len(population)
    additional_inds = []
    prev_nodes_ids = []
    for ind in population:
        ind_subtrees = [node for node in ind.graph.nodes if node != ind.graph.root_node]
        subtrees = [OptGraph(deepcopy(node.ordered_subnodes_hierarchy())) for node in ind_subtrees if
                    is_fitted_subtree(node, prev_nodes_ids)]
        additional_inds += subtrees
        prev_nodes_ids += [subtree.root_node.descriptive_id for subtree in subtrees]
        for add_ind in additional_inds:
            add_ind.parent_operators.append(
                ParentOperator(operator_type='regularization',
                               operator_name='decremental_regularization',
                               parent_individuals=[ind]))

    additional_inds = [ind for ind in additional_inds if constraint_function(ind, params)]

    is_multi_obj = isinstance(population[0].fitness, MultiObjFitness)
    if additional_inds:
        evaluate_individuals(additional_inds, objective_function, params,
                             is_multi_obj, timer=timer)

    if additional_inds and len(additional_inds) > size:
        additional_inds = sorted(additional_inds, key=lambda ind: ind.fitness)[:size]

    return additional_inds


def is_fitted_subtree(node: Any, prev_nodes_ids: List[Any]) -> bool:
    return node.nodes_from and node.descriptive_id not in prev_nodes_ids and node.fitted_model
