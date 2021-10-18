from typing import Any, Dict

from ..interfaces.serializable import CLASS_PATH_KEY, DELIMITER, Serializable


class EnumSerializer(Serializable):

    def to_json(self):
        return {
            "value": self.value,
            CLASS_PATH_KEY: f'{self.__module__}{DELIMITER}{self.__class__.__qualname__}'
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return cls(json_obj["value"])
