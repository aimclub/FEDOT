from enum import Enum
from importlib import import_module
from inspect import isclass, isfunction, ismethod
from typing import Any, Dict
from uuid import UUID

from .interfaces.serializable import Serializable

SIMPLE_OBJECT_INIT_DATA = '_init_data'
MODULE_X_NAME_DELIMITER = '/'
CLASS_PATH_KEY = '_class_path'


def dump_path_to_obj(obj: object) -> Dict[str, str]:
    """
    Dumps the full path (module + name) to the input object into the dict

    :param obj: object which path should be resolved (class, function or method)

    :return dict[str, str] with path to the object
    """
    if isclass(obj) or isfunction(obj) or ismethod(obj):
        obj_name = obj.__qualname__
    else:
        obj_name = obj.__class__.__qualname__

    obj_module = obj.__module__
    return {
        CLASS_PATH_KEY: f'{obj_module}{MODULE_X_NAME_DELIMITER}{obj_name}'
    }


def encoder(obj: Any) -> Dict[str, Any]:
    """
    Serves as 'default' parameter in json.dumps(...)

    :param obj: object to be encoded (class, function or method)

    :return dict[str, Any] which is in fact json object
    """
    if isinstance(obj, Serializable):
        return obj.to_json()
    elif isinstance(obj, UUID):
        return {
            SIMPLE_OBJECT_INIT_DATA: {'hex': obj.hex},
            **dump_path_to_obj(obj)
        }
    elif isinstance(obj, Enum):
        return {
            SIMPLE_OBJECT_INIT_DATA: obj.value,
            **dump_path_to_obj(obj)
        }
    elif isfunction(obj) or ismethod(obj):
        return dump_path_to_obj(obj)

    print(f'obj={obj} of type {type(obj)} can\'t be serialized.')
    return {}


def _get_class(class_path: str) -> Any:
    """
    Gets the object type from the class_path

    :param class_path: full path (module + name) of the class

    :return class, function or method type
    """
    module_name, class_name = class_path.split(MODULE_X_NAME_DELIMITER)
    obj_cls = import_module(module_name)
    for sub in class_name.split('.'):
        obj_cls = getattr(obj_cls, sub)
    return obj_cls


def decoder(json_obj: Dict[str, Any]) -> Any:
    """
    Serves as "object_hook" parameter in json.loads(...)

    :param json_obj: dict[str, Any] to be decoded into Python class, function or
        method object only if it has some special fields

    :return Python class, function or method object OR input if it's just a regular dict
    """
    if CLASS_PATH_KEY in json_obj:
        obj_cls = _get_class(json_obj[CLASS_PATH_KEY])
        del json_obj[CLASS_PATH_KEY]
        if isclass(obj_cls) and issubclass(obj_cls, Serializable):
            return obj_cls.from_json(json_obj)
        elif isfunction(obj_cls) or ismethod(obj_cls):
            return obj_cls
        elif SIMPLE_OBJECT_INIT_DATA in json_obj:
            init_data = json_obj[SIMPLE_OBJECT_INIT_DATA]
            if type(init_data) is dict:
                return obj_cls(**init_data)
            return obj_cls(init_data)
        raise TypeError(f'Parsed obj_cls={obj_cls} is not serializable, but should be')
    return json_obj
