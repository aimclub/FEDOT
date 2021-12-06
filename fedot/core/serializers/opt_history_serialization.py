import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Type

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import OptHistory

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from .any_serialization import any_from_json


def _convert_individuals_opt_graphs_to_templates(individuals: List[List['Individual']]):
    adapter = PipelineAdapter()
    for ind in list(itertools.chain(*individuals)):
        for parent_op in ind.parent_operators:
            parent_op.parent_objects = [
                adapter.restore_as_template(parent_obj)
                for parent_obj in parent_op.parent_objects
            ]
    return individuals


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized_hist = any_from_json(cls, json_obj)
    deserialized_hist.individuals = _convert_individuals_opt_graphs_to_templates(deserialized_hist.individuals)
    deserialized_hist.archive_history = _convert_individuals_opt_graphs_to_templates(deserialized_hist.archive_history)
    return deserialized_hist
