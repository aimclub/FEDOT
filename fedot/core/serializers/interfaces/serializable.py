from enum import Enum
from inspect import isclass
from typing import Any, Dict, Type, Union
from uuid import UUID

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.opt_history import OptHistory, ParentOperator
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.serializers.any_serialization import ClassOrFuncObject, any_from_json, any_to_json
from fedot.core.serializers.enum_serialization import enum_from_json, enum_to_json
from fedot.core.serializers.graph_node_serialization import graph_node_to_json
from fedot.core.serializers.graph_serialization import graph_from_json, graph_to_json
from fedot.core.serializers.operation_serialization import operation_to_json
from fedot.core.serializers.opt_history_serialization import opt_history_from_json
from fedot.core.serializers.parent_operator_serialization import parent_operator_to_json
from fedot.core.serializers.pipeline_template_serialization import pipeline_template_to_json
from fedot.core.serializers.uuid_serialization import uuid_from_json, uuid_to_json


class Serializer:
    _TO_JSON = 'to_json'
    _FROM_JSON = 'from_json'

    _processors_by_type = {
        GraphNode: {_TO_JSON: graph_node_to_json, _FROM_JSON: any_from_json},
        Graph: {_TO_JSON: graph_to_json, _FROM_JSON: graph_from_json},
        Operation: {_TO_JSON: operation_to_json, _FROM_JSON: any_from_json},
        OptHistory: {_TO_JSON: any_to_json, _FROM_JSON: opt_history_from_json},
        ParentOperator: {_TO_JSON: parent_operator_to_json, _FROM_JSON: any_from_json},
        PipelineTemplate: {_TO_JSON: pipeline_template_to_json, _FROM_JSON: any_from_json},
        UUID: {_TO_JSON: uuid_to_json, _FROM_JSON: uuid_from_json},
        Enum: {_TO_JSON: enum_to_json, _FROM_JSON: enum_from_json}
    }

    @staticmethod
    def is_serializable(obj: Union[ClassOrFuncObject, Type[ClassOrFuncObject]]) -> bool:
        types = tuple(Serializer._processors_by_type)
        if isclass(obj):
            return issubclass(obj, types)
        return isinstance(obj, types)

    @staticmethod
    def to_json(obj: ClassOrFuncObject) -> Dict[str, Any]:
        return Serializer._processors_by_type[type(obj)][Serializer._TO_JSON](obj)

    @staticmethod
    def from_json(cls: Type[ClassOrFuncObject], json_obj: Dict[str, Any]) -> ClassOrFuncObject:
        return Serializer._processors_by_type[cls][Serializer._FROM_JSON](cls, json_obj)
