import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Type

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import OptHistory

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from . import any_from_json


def _convert_parent_objects_ids_to_templates(individuals: List[List['Individual']]) -> List[List['Individual']]:
    for ind in list(itertools.chain(*individuals)):
        ind.graph.uid = ind.graph._serialization_id
        for parent_op in ind.parent_operators:
            for idx, parent_obj_id in enumerate(parent_op.parent_objects):
                for _ind in list(itertools.chain(*individuals)):
                    if parent_obj_id == _ind.graph._serialization_id:
                        parent_op.parent_objects[idx] = _ind
                        break
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    deserialized.individuals = _convert_parent_objects_ids_to_templates(deserialized.individuals)
    deserialized.archive_history = _convert_parent_objects_ids_to_templates(deserialized.archive_history)
    return deserialized
