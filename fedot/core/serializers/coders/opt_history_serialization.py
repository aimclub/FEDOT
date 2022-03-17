import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Type

from fedot.core.optimisers.opt_history import OptHistory

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from . import any_from_json


def _convert_parent_individuals(individuals: List[List['Individual']]) -> List[List['Individual']]:
    # TODO optimize quadruple for cycle
    for ind in list(itertools.chain(*individuals)):
        for parent_op in ind.parent_operators:
            for idx, parent_ind_uid in enumerate(parent_op.parent_individuals):
                for _ind in list(itertools.chain(*individuals)):
                    if parent_ind_uid == _ind.uid:
                        parent_op.parent_individuals[idx] = _ind
                        break
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    deserialized.individuals = _convert_parent_individuals(deserialized.individuals)
    deserialized.archive_history = _convert_parent_individuals(deserialized.archive_history)
    return deserialized
