import itertools
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from .interfaces.serializable import Serializable


class OptHistorySerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        return super().to_json()

    @staticmethod
    def __convert_individuals_pipelines_to_templates(individuals: List[List['Individual']]):
        from fedot.core.pipelines.template import PipelineTemplate

        for ind in list(itertools.chain(*individuals)):
            for parent_op in ind.parent_operators:
                parent_op.parent_objects = [
                    PipelineTemplate(parent_obj)
                    for parent_obj in parent_op.parent_objects
                ]
        return individuals

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        deserialized_hist = super().from_json(json_obj)
        deserialized_hist.individuals = cls.__convert_individuals_pipelines_to_templates(deserialized_hist.individuals)
        deserialized_hist.archive_history = cls.__convert_individuals_pipelines_to_templates(
            deserialized_hist.archive_history
        )
        return deserialized_hist
