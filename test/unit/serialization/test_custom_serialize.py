import json
from copy import deepcopy

import pytest
from typing import Dict

from fedot.core.serializers.serializer import register_serializable, Serializer, CLASS_PATH_KEY


class DataEq:
    def __init__(self, data):
        self.data = data
        self._data = data

    def __eq__(self, other):
        return (self.data == other.data and
                self._data == other._data)


@register_serializable(add_save_load=True)
class DefaultSerializable(DataEq):
    pass


custom_id = '0042'


def encode_custom(obj):
    return {custom_id: obj.data}


def decode_custom(cls, _dict):
    return cls(data=_dict[custom_id])


@register_serializable(to_json=encode_custom, from_json=decode_custom, add_save_load=True)
class CustomSerializable(DataEq):
    pass


@register_serializable(add_save_load=True)
class CustomSerializableWithMethods(DataEq):
    pass

    def to_json(self) -> Dict:
        return {custom_id: self.data}

    @classmethod
    def from_json(cls, _dict):
        return CustomSerializableWithMethods(_dict[custom_id])


@pytest.mark.parametrize('obj', [DefaultSerializable(42),
                                 CustomSerializable(666),
                                 CustomSerializableWithMethods(777)])
def test_serializable(obj):
    dumped = json.dumps(obj, cls=Serializer)
    loaded = json.loads(dumped, cls=Serializer)

    assert loaded == deepcopy(obj)


@pytest.mark.parametrize('obj', [DefaultSerializable(42),
                                 CustomSerializable(666),
                                 CustomSerializableWithMethods(777)])
def test_default_save_load(obj):
    # test that have 'save' and 'load' methods added by default
    assert hasattr(obj, 'save')
    assert hasattr(obj, 'load')
    assert obj.__class__.load(obj.save()) == obj


@pytest.mark.parametrize('obj', [CustomSerializableWithMethods(777)])
def test_serializable_with_class_methods(obj):
    dumped_srz = json.dumps(obj, cls=Serializer)
    dumped_self = obj.to_json()

    decoded_self = obj.from_json(dumped_self)
    decoded_srz = json.loads(dumped_srz, cls=Serializer)

    assert decoded_self == decoded_srz == deepcopy(obj)


@pytest.mark.parametrize('obj', [CustomSerializable(666)])
def test_serializable_custom(obj):
    dumped_srz = json.dumps(obj, cls=Serializer)
    dumped_self = encode_custom(obj)

    decoded_self = decode_custom(obj.__class__, dumped_self)
    decoded_srz = json.loads(dumped_srz, cls=Serializer)

    assert decoded_self == decoded_srz == deepcopy(obj)
