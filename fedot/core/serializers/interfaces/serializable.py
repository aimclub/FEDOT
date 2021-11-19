import copy
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict

from fedot.core.serializers import json_helpers


class Serializable(ABC):
    '''
    Base abstract class for serialization/deserialization process.

    Serialization: stores every field except "log" and properties,
        also dumps path to serialized object
    Deserialization: creates object based on its input (args, kwargs) and then updates object with
        other fields if needed
    '''

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in vars(self).items() if k != 'log'},
            **json_helpers.dump_path_to_obj(self)
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
