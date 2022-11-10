import json
import os
from importlib import import_module
from inspect import isclass, isfunction, ismethod, signature
from json import JSONDecoder, JSONEncoder
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

# NB: at the end of module init happens registration of default class coders

INSTANCE_OR_CALLABLE = TypeVar('INSTANCE_OR_CALLABLE', object, Callable)
EncodeCallable = Callable[[INSTANCE_OR_CALLABLE], Dict[str, Any]]
DecodeCallable = Callable[[Type[INSTANCE_OR_CALLABLE], Dict[str, Any]], INSTANCE_OR_CALLABLE]


MODULE_X_NAME_DELIMITER = '/'
CLASS_PATH_KEY = '_class_path'

# Mapping between class paths for backward compatibility for renamed/moved classes
LEGACY_CLASS_PATHS = {
    'fedot.core.optimisers.gp_comp.individual/Individual':
        'fedot.core.optimisers.opt_history_objects.individual/Individual',
    'fedot.core.optimisers.gp_comp.individual/ParentOperator':
        'fedot.core.optimisers.opt_history_objects.parent_operator/ParentOperator',
    'fedot.core.optimisers.opt_history/OptHistory':
        'fedot.core.optimisers.opt_history_objects.opt_history/OptHistory',

    'fedot.core.dag.graph_node/GraphNode':
        'fedot.core.dag.linked_graph_node/LinkedGraphNode',
    'fedot.core.dag.graph_operator/GraphOperator':
        'fedot.core.dag.linked_graph/LinkedGraph',
    'fedot.core.dag.graph_operator/GraphOperator._empty_postprocess':
        'fedot.core.dag.linked_graph/LinkedGraph._empty_postprocess',
}


class Serializer(JSONEncoder, JSONDecoder):

    _to_json = 'to_json'
    _from_json = 'from_json'

    CODERS_BY_TYPE = {}

    __default_coders_initialized = False

    def __init__(self, *args, **kwargs):
        for base_class, coder_name in [(JSONEncoder, 'default'), (JSONDecoder, 'object_hook')]:
            base_kwargs = {k: kwargs[k] for k in kwargs.keys() & signature(base_class.__init__).parameters}
            base_kwargs[coder_name] = getattr(self, coder_name)
            base_class.__init__(self, **base_kwargs)

        if not self.__default_coders_initialized:
            Serializer._register_default_coders()
            self.__default_coders_initialized = True

    @staticmethod
    def _register_default_coders():
        from uuid import UUID

        from fedot.core.dag.graph import Graph
        from fedot.core.dag.linked_graph_node import LinkedGraphNode
        from fedot.core.optimisers.opt_history_objects.individual import Individual
        from fedot.core.optimisers.opt_history_objects.generation import Generation
        from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
        from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator
        from fedot.core.optimisers.fitness.fitness import Fitness
        from fedot.core.optimisers.objective.objective import Objective
        from fedot.core.utilities.data_structures import ComparableEnum

        from .any_serialization import any_from_json, any_to_json

        from .coders import (
            enum_from_json,
            enum_to_json,
            graph_from_json,
            graph_node_to_json,
            opt_history_from_json,
            opt_history_to_json,
            objective_from_json,
            parent_operator_from_json,
            parent_operator_to_json,
            uuid_from_json,
            uuid_to_json
        )

        _to_json = Serializer._to_json
        _from_json = Serializer._from_json
        basic_serialization = {_to_json: any_to_json, _from_json: any_from_json}

        default_coders = {
            Fitness: basic_serialization,
            Individual: basic_serialization,
            Generation: basic_serialization,
            LinkedGraphNode: {_to_json: graph_node_to_json, _from_json: any_from_json},
            Graph: {_to_json: any_to_json, _from_json: graph_from_json},
            OptHistory: {_to_json: opt_history_to_json, _from_json: opt_history_from_json},
            Objective: {_to_json: any_to_json, _from_json: objective_from_json},
            ParentOperator: {_to_json: parent_operator_to_json, _from_json: parent_operator_from_json},
            UUID: {_to_json: uuid_to_json, _from_json: uuid_from_json},
            ComparableEnum: {_to_json: enum_to_json, _from_json: enum_from_json},
        }
        Serializer.CODERS_BY_TYPE.update(default_coders)

    @staticmethod
    def register_coders(cls: Type[INSTANCE_OR_CALLABLE],
                        to_json: Optional[EncodeCallable[INSTANCE_OR_CALLABLE]] = None,
                        from_json: Optional[DecodeCallable[INSTANCE_OR_CALLABLE]] = None,
                        overwrite: bool = False,
                        ) -> Type[INSTANCE_OR_CALLABLE]:
        """Registers classes as json-serializable so that they can be used by `Serializer`.

        Supports 3 alternative usages:

        - Default serialization is used (functions `any_to_json` and `any_from_json`).
        - Custom serialization functions `to_json` & `from_json` are provided explicitly as arguments.
        - Custom serialization functions `to_json` & `from_json` are defined in the class.

        Args:
            cls: registered class
            to_json: custom encoding function that returns json Dict.
             Optional, if None then default is used.
            from_json: custom decoding function that returns class given a Dict.
             Optional, if None then default is used.
            overwrite: flag that allows to overwrite existing registered coders.
             False by default: method raises AttributeError in attempt to overwrite coders.

        Returns:
            cls: class that is registered in serializer
        """
        if cls is None:
            raise TypeError('Class must not be None.')
        from .any_serialization import any_from_json, any_to_json

        # get provided coders or coders defined in the class itself or default universal coders
        coders = {Serializer._to_json: to_json or getattr(cls, Serializer._to_json, any_to_json),
                  Serializer._from_json: from_json or getattr(cls, Serializer._from_json, any_from_json)}

        if cls not in Serializer.CODERS_BY_TYPE or overwrite:
            Serializer.CODERS_BY_TYPE[cls] = coders
        else:
            raise AttributeError(f'Object {cls} already has serializer coders registered.')

        return cls

    @staticmethod
    def _get_field_checker(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> Callable[..., bool]:
        if isclass(obj):
            return issubclass
        return isinstance

    @staticmethod
    def _get_base_type(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> Optional[Type]:
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
            encoded = Serializer._get_coder_by_type(base_type, Serializer._to_json)(obj)
            if CLASS_PATH_KEY not in encoded:
                encoded.update(Serializer.dump_path_to_obj(obj))
            return encoded

        return JSONEncoder.default(self, obj)

    @staticmethod
    def _get_class(class_path: str) -> Type[INSTANCE_OR_CALLABLE]:
        """
        Gets the object type from the class_path

        :param class_path: full path (module + name) of the class

        :return: class, function or method type
        """
        class_path = LEGACY_CLASS_PATHS.get(class_path, class_path)
        module_name, class_name = class_path.split(MODULE_X_NAME_DELIMITER)
        obj_cls = import_module(module_name)
        for sub in class_name.split('.'):
            obj_cls = getattr(obj_cls, sub)
        return obj_cls

    @staticmethod
    def _is_bound_method(method: Callable) -> bool:
        return hasattr(method, '__self__')

    @staticmethod
    def object_hook(json_obj: Dict[str, Any]) -> Union[INSTANCE_OR_CALLABLE, dict]:
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
                coder = Serializer._get_coder_by_type(base_type, Serializer._from_json)
                # call with a right num of arguments
                if Serializer._is_bound_method(coder):
                    return coder(json_obj)
                else:
                    return coder(obj_cls, json_obj)
            elif isfunction(obj_cls) or ismethod(obj_cls):
                return obj_cls
            raise TypeError(f'Parsed obj_cls={obj_cls} is not serializable, but should be')
        return json_obj


def default_save(obj: Any, json_file_path: Optional[Union[str, os.PathLike]] = None) -> Optional[str]:
    """ Default save to json using Serializer """
    if json_file_path is None:
        return json.dumps(obj, indent=4, cls=Serializer)
    with open(json_file_path, mode='w') as json_file:
        json.dump(obj, json_file, indent=4, cls=Serializer)
        return None


def default_load(json_str_or_file_path: Union[str, os.PathLike]) -> Any:
    """ Default load from json using Serializer """
    def load_as_file_path():
        with open(json_str_or_file_path, mode='r') as json_file:
            return json.load(json_file, cls=Serializer)

    def load_as_json_str():
        return json.loads(json_str_or_file_path, cls=Serializer)

    if isinstance(json_str_or_file_path, os.PathLike):
        return load_as_file_path()

    try:
        return load_as_json_str()
    except json.JSONDecodeError:
        return load_as_file_path()


def register_serializable(cls: Optional[Type[INSTANCE_OR_CALLABLE]] = None,
                          to_json: Optional[EncodeCallable] = None,
                          from_json: Optional[DecodeCallable] = None,
                          overwrite: bool = False,
                          add_save_load: bool = False,
                          ) -> Type[INSTANCE_OR_CALLABLE]:
    """Decorator for registering classes as json-serializable.
    Relies on :py:class:`fedot.core.serializers.serializer.Serializer.register_coders`.
    Optionally adds `save` and `load` methods to the class for json (de)serialization.

    Args:
        cls: decorated class
        to_json: custom encoding function that returns json Dict.
         Optional, if None then default is used.
        from_json: custom decoding function that returns class given a Dict.
         Optional, if None then default is used.
        overwrite: flag that allows to overwrite existing registered coders.
         False by default: method raises AttributeError in attempt to overwrite coders.
        add_save_load: if True, then `save` and `load` methods are added to the class
         that (de)serialize the class (from)to the file using `Serializer`.
         See `default_save` & `default_load` functions.

    Returns:
        cls: class that is registered in serializer
    """
    _save = 'save'
    _load = 'load'

    def make_serializable(cls):
        Serializer.register_coders(cls, to_json, from_json, overwrite)
        if add_save_load:
            if hasattr(cls, _save) or hasattr(cls, _load):
                raise ValueError(f'Class {cls} already have `save` and/or `load` methods, can not overwrite them.')
            setattr(cls, _save, default_save)
            setattr(cls, _load, default_load)
        return cls

    # See if we're being called as @decorator() (with parens).
    if cls is None:
        # We're called with parens.
        return make_serializable

    # We're called as `@decorator` without parens.
    return make_serializable(cls)
