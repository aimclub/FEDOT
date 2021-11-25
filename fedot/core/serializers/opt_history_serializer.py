import itertools
from typing import Any, Dict

from fedot.core.serializers.interfaces.serializable import Serializable


class OptHistorySerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        return super().to_json()

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        from fedot.core.pipelines.template import PipelineTemplate
        deserialized_hist = super().from_json(json_obj)
        for ind in list(itertools.chain(*deserialized_hist.individuals)):
            for parent_op in ind.parent_operators:
                parent_op.parent_objects = [
                    PipelineTemplate(parent_obj)
                    for parent_obj in parent_op.parent_objects
                ]
        for ind in list(itertools.chain(*deserialized_hist.archive_history)):
            for parent_op in ind.parent_operators:
                parent_op.parent_objects = [
                    PipelineTemplate(parent_obj)
                    for parent_obj in parent_op.parent_objects
                ]
        return deserialized_hist
