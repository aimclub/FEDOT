from copy import deepcopy
from enum import Enum
from typing import (Optional,
                    List,
                    Any,
                    Callable,
                    Tuple)

import numpy as np


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


def regularized_population(reg_id: RegularizationTypesEnum, population: List[Any], requirements, metric: Callable,
                           chain_class: Any, size: Optional[int] = None) -> Tuple[List[Any], List[float]]:
    if reg_id == RegularizationTypesEnum.decremental:
        return decremental_regularization(population, requirements, metric, chain_class, size)
    elif reg_id == RegularizationTypesEnum.none:
        return ([], [])


def decremental_regularization(population: List[Any], requirements, metric: Callable, chain_class: Any,
                               size: Optional[int] = None) -> Tuple[List[Any], List[float]]:
    size = size if size else requirements.pop_size
    additional_inds = []
    prev_nodes_ids = []
    for ind in population:
        ind_subtrees = [node for node in ind.nodes if node != ind.root_node]
        subtrees = [deepcopy(chain_class(node.ordered_subnodes_hierarchy)) for node in ind_subtrees if
                    is_fitted_subtree(node, prev_nodes_ids)]
        additional_inds += subtrees
        prev_nodes_ids += [subtree.root_node.descriptive_id for subtree in subtrees]
    metrics = [metric(additional_ind) for additional_ind in additional_inds]
    if additional_inds and len(additional_inds) > size:
        sort_metrics_ids = np.argsort(metrics)
        metrics = [metrics[i] for i in sort_metrics_ids][:size]
        additional_inds = [additional_inds[i] for i in sort_metrics_ids][:size]
    return (additional_inds, metrics)


def is_fitted_subtree(node: Any, prev_nodes_ids: List[Any]) -> bool:
    return node.nodes_from and not node.descriptive_id in prev_nodes_ids and node.cache.actual_cached_state
