from typing import Any, Dict

from ..interfaces.serializable import (CLASS_PATH_KEY, DELIMITER, Serializable,
                                       dump_path_to_obj)


class EnumSerializer(Serializable):

    def to_json(self):
        return {
            'value': self.value,
            **dump_path_to_obj(self)
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return cls(json_obj['value'])
