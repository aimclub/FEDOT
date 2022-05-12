import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Type

from fedot.core.optimisers.opt_history import OptHistory

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from . import any_from_json


def _convert_parent_individuals(individuals: List[List['Individual']]) -> List[List['Individual']]:
    # get all individuals from all generations
    all_individuals = reduce(operator.concat, individuals)
    lookup_dict = {ind.uid: ind for ind in all_individuals}

    for ind in all_individuals:
        for parent_op in ind.parent_operators:
            for parent_ind_idx, parent_ind_uid in enumerate(parent_op.parent_individuals):
                parent_op.parent_individuals[parent_ind_idx] = lookup_dict.get(parent_ind_uid, None)
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    deserialized.individuals = _convert_parent_individuals(deserialized.individuals)
    deserialized.archive_history = _convert_parent_individuals(deserialized.archive_history)
    return deserialized
