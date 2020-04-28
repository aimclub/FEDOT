from copy import deepcopy, copy
from enum import Enum
from typing import (Optional,
                    List,
                    Any)


class RegularizationTypeEnum(Enum):
    none = 0
    decremental = 1


def regularized_population(reg_id, population, requirements, metric, chain_class, size=None) -> Optional[List[Any]]:
    if reg_id == RegularizationTypeEnum.decremental:
        return population + decremental_regularization(population, requirements, metric, chain_class, size)
    elif reg_id == RegularizationTypeEnum.none:
        return population

def decremental_regularization(population, requirements, metric, chain_class, size=None):
    size = size if size else requirements.pop_size
    additional_inds = []
    prev_nodes_ids = []
    for ind in population:
        ind_copy = deepcopy(ind)
        ind_copy_subtrees = [node for node in ind_copy.nodes if node != ind_copy.root_node]
        subtrees = [chain_class(node.subtree_nodes) for node in ind_copy_subtrees if
                    node.nodes_from and not node.descriptive_id in prev_nodes_ids and node.cache.actual_cached_model]
        additional_inds += subtrees
        prev_nodes_ids += [subtree.root_node.descriptive_id for subtree in subtrees]
    if additional_inds and len(additional_inds) > size:
        additional_inds = sorted(additional_inds, key=lambda chain: round(metric(chain), 3))[:size]
    return additional_inds
