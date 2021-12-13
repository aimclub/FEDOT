from copy import deepcopy
from inspect import signature
from typing import Any, Dict, Type

from .. import INSTANCE_OR_CALLABLE, Serializer


def any_to_json(obj: INSTANCE_OR_CALLABLE) -> Dict[str, Any]:
    return {
        **{k: v for k, v in vars(obj).items() if k != 'log'},
        **Serializer.dump_path_to_obj(obj)
    }


def any_from_json(cls: Type[INSTANCE_OR_CALLABLE], json_obj: Dict[str, Any]) -> INSTANCE_OR_CALLABLE:
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
