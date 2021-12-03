from copy import deepcopy
from inspect import signature
from typing import Any, Dict, Type, TypeVar

from .json_helpers import dump_path_to_obj

ClassOrFuncObject = TypeVar('ClassOrFuncObject')


def any_to_json(object: ClassOrFuncObject) -> Dict[str, Any]:
    return {
        **{k: v for k, v in vars(object).items() if k != 'log'},
        **dump_path_to_obj(object)
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
