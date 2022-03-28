from importlib import import_module
from inspect import isclass, isfunction, ismethod, signature
from json import JSONDecoder, JSONEncoder
from typing import Any, Callable, Dict, Type, TypeVar, Union

MODULE_X_NAME_DELIMITER = '/'
INSTANCE_OR_CALLABLE = TypeVar('INSTANCE_OR_CALLABLE', object, Callable)
CLASS_PATH_KEY = '_class_path'


class Serializer(JSONEncoder, JSONDecoder):

    _to_json = 'to_json'
    _from_json = 'from_json'

    CODERS_BY_TYPE = {}

    def __init__(self, *args, **kwargs):
        for base_class, coder_name in [(JSONEncoder, 'default'), (JSONDecoder, 'object_hook')]:
            base_kwargs = {k: kwargs[k] for k in kwargs.keys() & signature(base_class.__init__).parameters}
            base_kwargs[coder_name] = getattr(self, coder_name)
            base_class.__init__(self, **base_kwargs)

        if not Serializer.CODERS_BY_TYPE:
            from uuid import UUID

            from fedot.core.dag.graph import Graph
            from fedot.core.dag.graph_node import GraphNode
            from fedot.core.operations.operation import Operation
            from fedot.core.optimisers.gp_comp.individual import Individual
            from fedot.core.optimisers.graph import OptGraph, OptNode
            from fedot.core.optimisers.opt_history import OptHistory, ParentOperator
            from fedot.core.utilities.data_structures import ComparableEnum

            from .coders import (
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
                uuid_from_json,
                uuid_to_json
            )

            _to_json = Serializer._to_json
            _from_json = Serializer._from_json
            basic_serialization = {_to_json: any_to_json, _from_json: any_from_json}
            Serializer.CODERS_BY_TYPE = {
                Individual: basic_serialization,
                GraphNode: {_to_json: graph_node_to_json, _from_json: any_from_json},
                Graph: {_to_json: graph_to_json, _from_json: graph_from_json},
                Operation: {_to_json: operation_to_json, _from_json: any_from_json},
                OptHistory: {_to_json: any_to_json, _from_json: opt_history_from_json},
                ParentOperator: {_to_json: parent_operator_to_json, _from_json: any_from_json},
                UUID: {_to_json: uuid_to_json, _from_json: uuid_from_json},
                ComparableEnum: {_to_json: enum_to_json, _from_json: enum_from_json},
            }
            Serializer.CODERS_BY_TYPE.update({
                OptNode: Serializer.CODERS_BY_TYPE[GraphNode],
                OptGraph: Serializer.CODERS_BY_TYPE[Graph],
            })

    @staticmethod
    def _get_field_checker(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> Callable[..., bool]:
        if isclass(obj):
            return issubclass
        return isinstance

    @staticmethod
    def _get_base_type(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> int:
        contains = Serializer._get_field_checker(obj)
        for k_type in Serializer.CODERS_BY_TYPE:
            if contains(obj, k_type):
                return k_type
        return None

    @staticmethod
    def _get_coder_by_type(coder_type: Type, coder_aim: str):
        return Serializer.CODERS_BY_TYPE[coder_type][coder_aim]

    @staticmethod
    def dump_path_to_obj(obj: INSTANCE_OR_CALLABLE) -> Dict[str, str]:
        """
        Dumps the full path (module + name) to the input object into the dict

        :param obj: object which path should be resolved (class, function or method)

        :return: dict[str, str] with path to the object
        """
        if isclass(obj) or isfunction(obj) or ismethod(obj):
            obj_name = obj.__qualname__
        else:
            obj_name = obj.__class__.__qualname__

        if getattr(obj, '__module__', None) is not None:
            obj_module = obj.__module__
        else:
            obj_module = obj.__class__.__module__
        return {
            CLASS_PATH_KEY: f'{obj_module}{MODULE_X_NAME_DELIMITER}{obj_name}'
        }

    def default(self, obj: INSTANCE_OR_CALLABLE) -> Dict[str, Any]:
        """
        Tries to encode objects that are not simply json-encodable to JSON-object

        :param obj: object to be encoded (class, function or method)

        :return: dict[str, Any] which is in fact json object
        """
        if isfunction(obj) or ismethod(obj):
            return Serializer.dump_path_to_obj(obj)
        base_type = Serializer._get_base_type(obj)
        if base_type is not None:
            return Serializer._get_coder_by_type(base_type, Serializer._to_json)(obj)

        return JSONEncoder.default(self, obj)

    @staticmethod
    def _get_class(class_path: str) -> Type[INSTANCE_OR_CALLABLE]:
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

    def object_hook(self, json_obj: Dict[str, Any]) -> Union[INSTANCE_OR_CALLABLE, dict]:
        """
        Decodes every JSON-object to python class/func object or just returns dict

        :param json_obj: dict[str, Any] to be decoded into Python class, function or
            method object only if it has some special fields

        :return: Python class, function or method object OR input if it's just a regular dict
        """
        if CLASS_PATH_KEY in json_obj:
            obj_cls = Serializer._get_class(json_obj[CLASS_PATH_KEY])
            del json_obj[CLASS_PATH_KEY]
            base_type = Serializer._get_base_type(obj_cls)
            if isclass(obj_cls) and base_type is not None:
                return Serializer._get_coder_by_type(base_type, Serializer._from_json)(obj_cls, json_obj)
            elif isfunction(obj_cls) or ismethod(obj_cls):
                return obj_cls
            raise TypeError(f'Parsed obj_cls={obj_cls} is not serializable, but should be')
        return json_obj
