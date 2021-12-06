from importlib import import_module
from inspect import isclass, isfunction, ismethod
from json import JSONDecoder, JSONEncoder
from typing import Any, Dict, Type, TypeVar, Union
from uuid import UUID

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.opt_history import OptHistory, ParentOperator
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.utils import ComparableEnum

MODULE_X_NAME_DELIMITER = '/'
CLASS_OR_FUNC_OBJECT = TypeVar('CLASS_OR_FUNC_OBJECT')
CLASS_PATH_KEY = '_class_path'


class Serializer(JSONEncoder, JSONDecoder):

    _to_json = 'to_json'
    _from_json = 'from_json'

    PROCESSORS_BY_TYPE = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from .encoders import (
            any_from_json,
            any_to_json,
            enum_from_json,
            enum_to_json,
            graph_from_json,
            graph_node_to_json,
            graph_to_json,
            operation_to_json,
            opt_history_from_json,
            parent_operator_to_json,
            pipeline_template_to_json,
            uuid_from_json,
            uuid_to_json
        )

        _to_json = Serializer._to_json
        _from_json = Serializer._from_json

        if not Serializer.PROCESSORS_BY_TYPE:
            Serializer.PROCESSORS_BY_TYPE = {
                GraphNode: {_to_json: graph_node_to_json, _from_json: any_from_json},
                Graph: {_to_json: graph_to_json, _from_json: graph_from_json},
                Operation: {_to_json: operation_to_json, _from_json: any_from_json},
                OptHistory: {_to_json: any_to_json, _from_json: opt_history_from_json},
                ParentOperator: {
                    _to_json: parent_operator_to_json,
                    _from_json: any_from_json
                },
                PipelineTemplate: {
                    _to_json: pipeline_template_to_json,
                    _from_json: any_from_json
                },
                UUID: {_to_json: uuid_to_json, _from_json: uuid_from_json},
                ComparableEnum: {_to_json: enum_to_json, _from_json: enum_from_json}
            }

    @staticmethod
    def is_serializable(obj: Union[CLASS_OR_FUNC_OBJECT, Type[CLASS_OR_FUNC_OBJECT]]) -> bool:
        types = Serializer.PROCESSORS_BY_TYPE.keys()
        if isclass(obj):
            return obj in types
        return type(obj) in types

    @staticmethod
    def dump_path_to_obj(obj: CLASS_OR_FUNC_OBJECT) -> Dict[str, str]:
        """
        Dumps the full path (module + name) to the input object into the dict

        :param obj: object which path should be resolved (class, function or method)

        :return: dict[str, str] with path to the object
        """
        if isclass(obj) or isfunction(obj) or ismethod(obj):
            obj_name = obj.__qualname__
        else:
            obj_name = obj.__class__.__qualname__

        obj_module = obj.__module__
        return {
            CLASS_PATH_KEY: f'{obj_module}{MODULE_X_NAME_DELIMITER}{obj_name}'
        }

    def default(self, obj: CLASS_OR_FUNC_OBJECT) -> Dict[str, Any]:
        """
        Tries to encode objects that are not simply json-encodable to JSON-object

        :param obj: object to be encoded (class, function or method)

        :return: dict[str, Any] which is in fact json object
        """
        if isfunction(obj) or ismethod(obj):
            return Serializer.dump_path_to_obj(obj)
        elif Serializer.is_serializable(obj):
            return Serializer.PROCESSORS_BY_TYPE[type(obj)][Serializer._to_json](obj)

        return JSONEncoder.default(self, obj)

    @staticmethod
    def _get_class(class_path: str) -> Type[CLASS_OR_FUNC_OBJECT]:
        """
        Gets the object type from the class_path

        :param class_path: full path (module + name) of the class

        :return: class, function or method type
        """
        module_name, class_name = class_path.split(MODULE_X_NAME_DELIMITER)
        obj_cls = import_module(module_name)
        for sub in class_name.split('.'):
            obj_cls = getattr(obj_cls, sub)
        return obj_cls

    def object_hook(self, json_obj: Dict[str, Any]) -> Union[CLASS_OR_FUNC_OBJECT, dict]:
        """
        Decodes every JSON-object to python class/func object or just returns dict

        :param json_obj: dict[str, Any] to be decoded into Python class, function or
            method object only if it has some special fields

        :return: Python class, function or method object OR input if it's just a regular dict
        """
        if CLASS_PATH_KEY in json_obj:
            obj_cls = Serializer._get_class(json_obj[CLASS_PATH_KEY])
            del json_obj[CLASS_PATH_KEY]
            if isclass(obj_cls) and Serializer.is_serializable(obj_cls):
                return Serializer.PROCESSORS_BY_TYPE[obj_cls][Serializer._from_json](obj_cls, json_obj)
            elif isfunction(obj_cls) or ismethod(obj_cls):
                return obj_cls
            raise TypeError(f'Parsed obj_cls={obj_cls} is not serializable, but should be')
        return json_obj
