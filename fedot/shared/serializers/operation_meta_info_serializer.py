from inspect import isclass
from typing import Any, Dict

from ..interfaces.serializable import (CLASS_PATH_KEY, DELIMITER, Serializable,
                                       dump_path_to_obj)
from .json_helpers import _get_class


class OperationMetaInfoSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        basic_serialization = super().to_json()
        strategy = basic_serialization['supported_strategies']
        if isclass(strategy):
            basic_serialization['supported_strategies'] = dump_path_to_obj(strategy)[CLASS_PATH_KEY]
        return basic_serialization

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        json_obj['supported_strategies'] = _get_class(json_obj['supported_strategies'])
        return super().from_json(json_obj)
