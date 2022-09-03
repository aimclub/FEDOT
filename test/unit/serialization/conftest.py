from uuid import UUID

import pytest

from fedot.core.serializers import Serializer
from fedot.core.serializers.coders import (
    any_from_json,
    any_to_json,
    enum_from_json,
    enum_to_json,
    graph_from_json,
    graph_node_to_json,
    operation_to_json,
    uuid_from_json,
    uuid_to_json
)
from .mocks.serialization_mocks import MockGraph, MockNode, MockOperation
from .shared_data import TestClass, TestEnum, TestSerializableClass


@pytest.fixture
def get_class_fixture(monkeypatch):
    def mock_get_class(obj_type: type):
        return obj_type

    monkeypatch.setattr(
        f'{Serializer._get_class.__module__}.{Serializer._get_class.__qualname__}',
        mock_get_class
    )


@pytest.fixture
def mock_classes_fixture(monkeypatch):
    _to_json = Serializer._to_json
    _from_json = Serializer._from_json
    monkeypatch.setattr(Serializer, 'CODERS_BY_TYPE', {
        MockNode: {_to_json: graph_node_to_json, _from_json: any_from_json},
        MockGraph: {_to_json: any_to_json, _from_json: graph_from_json},
        MockOperation: {_to_json: operation_to_json, _from_json: any_from_json},
        UUID: {_to_json: uuid_to_json, _from_json: uuid_from_json},
        TestEnum: {_to_json: enum_to_json, _from_json: enum_from_json},
        TestClass: {_to_json: any_to_json, _from_json: any_from_json},
        TestSerializableClass: {_to_json: any_to_json, _from_json: any_from_json}
    })
