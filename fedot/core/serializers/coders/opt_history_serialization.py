import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Type

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import OptHistory

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from . import any_from_json


def _convert_individuals_opt_graphs_to_templates(individuals: List[List['Individual']]):
    # adapter = PipelineAdapter()
    for ind in list(itertools.chain(*individuals)):
        for parent_op in ind.parent_operators:
            for idx in range(len(parent_op.parent_objects)):
                cur = parent_op.parent_objects[idx]
                for _ind in list(itertools.chain(*individuals)):
                    if cur == _ind.graph.uid:
                        parent_op.parent_objects[idx] = _ind.graph
                        break
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    json_obj['individuals'] = _convert_individuals_opt_graphs_to_templates(json_obj['individuals'])
    json_obj['archive_history'] = _convert_individuals_opt_graphs_to_templates(json_obj['archive_history'])
    deserialized_hist = any_from_json(cls, json_obj)
    return deserialized_hist
