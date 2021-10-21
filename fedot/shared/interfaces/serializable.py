import copy
from abc import ABC, abstractmethod
from inspect import isclass, isfunction, ismethod, signature
from typing import Any, Dict

DELIMITER = '/'
CLASS_PATH_KEY = '_class_path'

def dump_path_to_obj(obj: object) -> Dict[str, str]:
    if isclass(obj) or isfunction(obj) or ismethod(obj):
        obj_name = obj.__qualname__
    else:
        obj_name = obj.__class__.__qualname__
    if getattr(obj, '__module__', None) is None:
        obj_module = obj.__class__.__module__
    else:
        obj_module = obj.__module__
    return {
        CLASS_PATH_KEY: f'{obj_module}{DELIMITER}{obj_name}'
    }

class Serializable(ABC):

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        return {
            **vars(self),
            **dump_path_to_obj(self)
        }

    @classmethod
    @abstractmethod
    def from_json(cls, json_obj: Dict[str, Any]) -> Any:
        cls_parameters = signature(cls.__init__).parameters
        if 'kwargs' not in cls_parameters:
            init_data = {
                k: v
                for k, v in json_obj.items()
                if k in cls_parameters
            }
            obj = cls(**init_data)
            vars(obj).update({
                k: json_obj[k]
                for k in json_obj.keys() ^ init_data.keys()
            })
        else:
            init_data = copy.deepcopy(json_obj)
            obj = cls(**init_data)
            vars(obj).update(json_obj)
        return obj
