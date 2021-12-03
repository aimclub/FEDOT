from copy import deepcopy
from inspect import signature
from typing import Any, Dict, Type, TypeVar

from . import json_helpers

ClassOrFuncObject = TypeVar('ClassOrFuncObject')


def any_to_json(obj: ClassOrFuncObject) -> Dict[str, Any]:
    return {
        **{k: v for k, v in vars(obj).items() if k != 'log'},
        **json_helpers.dump_path_to_obj(obj)
    }


def any_from_json(cls: Type[ClassOrFuncObject], json_obj: Dict[str, Any]) -> ClassOrFuncObject:
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
        init_data = deepcopy(json_obj)
        obj = cls(**init_data)
        vars(obj).update(json_obj)
    return obj
