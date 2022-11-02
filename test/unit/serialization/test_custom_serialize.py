import pytest
from typing import Dict

from fedot.core.serializers.serializer import register_serializable


@register_serializable(add_save_load=True)
class DefaultSerializable:
    def __init__(self, data):
        self.data = data
        self._data = data


custom_id = '0042'


def encode_custom(obj):
    return {custom_id: obj.data}


def decode_custom(cls, _dict):
    return cls(data=_dict[custom_id])


@register_serializable(to_json=encode_custom, from_json=decode_custom, add_save_load=True)
class CustomSerializable:
    def __init__(self, data):
        self.data = data
        self._data = data


@register_serializable(add_save_load=True)
class CustomSerializableWithMethods:
    def __init__(self, data):
        self.data = data
        self._data = data

    def to_json(self) -> Dict:
        return {custom_id: self.data}

    @classmethod
    def from_json(cls, _dict):
        return CustomSerializableWithMethods(_dict[custom_id])


@pytest.mark.parametrize('obj', [DefaultSerializable(42),
                                 CustomSerializable(666),
                                 CustomSerializableWithMethods(777)])
def test_serializable(obj):

    # check custom id

    pass


def test_serializable_custom(obj):
    # check custom id

    pass
