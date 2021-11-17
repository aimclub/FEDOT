import copy
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict

try:
    from fedot.serializers import json_helpers
except ImportError:
    import sys
    json_helpers = sys.modules['fedot.serializers.json_helpers']


class Serializable(ABC):

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        useless_fields = ['log', 'operation_templates']
        return {
            **{k: v for k, v in vars(self).items() if k not in useless_fields},
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
