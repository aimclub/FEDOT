from uuid import UUID

import pytest
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.operations.operation import Operation
# from fedot.core.optimisers.opt_history import OptHistory, ParentOperator
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.serializers import json_helpers
from fedot.core.serializers.any_serialization import any_from_json, any_to_json
from fedot.core.serializers.enum_serialization import enum_from_json, enum_to_json
from fedot.core.serializers.graph_node_serialization import graph_node_to_json
from fedot.core.serializers.graph_serialization import graph_from_json, graph_to_json
from fedot.core.serializers.interfaces.serializable import Serializer
from fedot.core.serializers.operation_serialization import operation_to_json
from fedot.core.serializers.pipeline_template_serialization import pipeline_template_to_json
from fedot.core.serializers.uuid_serialization import uuid_from_json, uuid_to_json
from fedot.core.utils import ComparableEnum

from ..mocks.serialization_mocks import MockGraph, MockNode, MockOperation, MockPipelineTemplate
from ..shared_data import TestEnum, TestSerializableClass


@pytest.fixture
def _get_class_fixture(monkeypatch):
    def mock_get_class(obj_type: type):
        return obj_type

    monkeypatch.setattr(
        f'{json_helpers._get_class.__module__}.{json_helpers._get_class.__qualname__}',
        mock_get_class
    )


@pytest.fixture
def _mock_classes_fixture(monkeypatch):
    _TO_JSON = Serializer._TO_JSON
    _FROM_JSON = Serializer._FROM_JSON
    monkeypatch.setattr(Serializer, '_processors_by_type', {
        MockNode: {_TO_JSON: graph_node_to_json, _FROM_JSON: any_from_json},
        MockGraph: {_TO_JSON: graph_to_json, _FROM_JSON: graph_from_json},
        MockOperation: {_TO_JSON: operation_to_json, _FROM_JSON: any_from_json},
        # OptHistory: {_TO_JSON: any_to_json, _FROM_JSON: opt_history_from_json},
        # ParentOperator: {_TO_JSON: parent_operator_to_json, _FROM_JSON: any_from_json},
        MockPipelineTemplate: {_TO_JSON: pipeline_template_to_json, _FROM_JSON: any_from_json},
        UUID: {_TO_JSON: uuid_to_json, _FROM_JSON: uuid_from_json},
        TestEnum: {_TO_JSON: enum_to_json, _FROM_JSON: enum_from_json},
        TestSerializableClass: {_TO_JSON: any_to_json, _FROM_JSON: any_from_json}
    })
