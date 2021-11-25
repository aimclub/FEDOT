from typing import Any, Dict

from fedot.core.pipelines.template import PipelineTemplate

from .interfaces.serializable import Serializable


class ParentOperatorSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        serialized_parent = super().to_json()
        serialized_parent['parent_objects'] = [
            parent_obj.link_to_empty_pipeline
            for parent_obj in serialized_parent['parent_objects']
        ]
        return serialized_parent

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
