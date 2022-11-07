from copy import deepcopy
from inspect import signature
from typing import Any, Dict, Type, TypeVar, Callable


INSTANCE_OR_CALLABLE = TypeVar('INSTANCE_OR_CALLABLE', object, Callable)


def any_to_json(obj: INSTANCE_OR_CALLABLE) -> Dict[str, Any]:
    return {k: v for k, v in sorted(vars(obj).items()) if not _is_log_var(k)}


def any_from_json(cls: Type[INSTANCE_OR_CALLABLE], json_obj: Dict[str, Any]) -> INSTANCE_OR_CALLABLE:
    obj = cls.__new__(cls)
    vars(obj).update(json_obj)
    return obj


def _is_log_var(varname: str) -> bool:
    return varname.strip('_').startswith('log')
