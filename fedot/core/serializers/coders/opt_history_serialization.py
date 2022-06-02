import operator
from functools import reduce
from typing import Any, Dict, List, Type

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory


from . import any_from_json


def _convert_parent_individuals(individuals: List[List['Individual']]) -> List[List['Individual']]:
    # get all individuals from all generations
    if individuals:
        all_individuals = reduce(operator.concat, individuals)
        lookup_dict = {ind.uid: ind for ind in all_individuals}

        for ind in all_individuals:
            for parent_op in ind.parent_operators:
                for parent_ind_idx, parent_ind_uid in enumerate(parent_op.parent_individuals):
                    parent_ind = lookup_dict.get(parent_ind_uid, None)
                    if parent_ind is None:
                        parent_ind = Individual(graph=OptGraph())
                    parent_op.parent_individuals[parent_ind_idx] = parent_ind
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    deserialized.individuals = _convert_parent_individuals(deserialized.individuals)
    deserialized.archive_history = _convert_parent_individuals(deserialized.archive_history)
    return deserialized
