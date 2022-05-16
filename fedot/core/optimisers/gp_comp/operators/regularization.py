from copy import deepcopy
from typing import Any, List, Optional

from fedot.core.composer.constraint import constraint_function
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.evaluation import EvaluationDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


def regularized_population(reg_type: RegularizationTypesEnum, population: PopulationT,
                           evaluator: EvaluationDispatcher,
                           params: GraphGenerationParams,
                           size: Optional[int] = None) -> PopulationT:
    if reg_type is RegularizationTypesEnum.decremental:
        return decremental_regularization(population, evaluator, params, size)
    elif reg_type is RegularizationTypesEnum.none:
        return population
    else:
        raise ValueError(f'Required regularization type not found: {type}')


def decremental_regularization(population: PopulationT,
                               evaluator: EvaluationDispatcher,
                               params: GraphGenerationParams,
                               size: Optional[int] = None) -> PopulationT:
    size = size or len(population)
    additional_inds = []
    prev_nodes_ids = set()
    for ind in population:
        prev_nodes_ids.add(ind.graph.root_node.descriptive_id)
        subtree_inds = [Individual(OptGraph(deepcopy(node.ordered_subnodes_hierarchy())))
                        for node in ind.graph.nodes
                        if is_fitted_subtree(node) and node.descriptive_id not in prev_nodes_ids]

        parent_operator = ParentOperator(operator_type='regularization',
                                         operator_name='decremental_regularization',
                                         parent_individuals=[ind])
        for add_ind in subtree_inds:
            add_ind.parent_operators.append(parent_operator)
        additional_inds.extend(subtree_inds)
        prev_nodes_ids.update(subtree.graph.root_node.descriptive_id for subtree in subtree_inds)

    additional_inds = [ind for ind in additional_inds if constraint_function(ind.graph, params)]

    evaluator(additional_inds)
    additional_inds.extend(population)
    if len(additional_inds) > size:
        additional_inds = sorted(additional_inds, key=lambda ind: ind.fitness)[:size]

    return additional_inds


def is_fitted_subtree(node: Any) -> bool:
    return node.nodes_from and node.fitted_model
