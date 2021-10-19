import copy
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from importlib import import_module
from inspect import signature
from typing import Any, Dict

DELIMITER = '/'
CLASS_PATH_KEY = '_class_path'


class Serializable(ABC):

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        return {
            **vars(self),
            CLASS_PATH_KEY: f'{self.__module__}{DELIMITER}{self.__class__.__qualname__}'
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
