from enum import Enum

from fedot.shared import BasicSerializer
from fedot.shared.serializers.graph_node_serializer import GraphNodeSerializer
from fedot.shared.serializers.graph_serializer import GraphSerializer
from fedot.shared.serializers.operation_serializer import OperationSerializer

TEST_UUID = '41d79d06c3d8478f89e7d1008c96a864'
TEST_INPUT_MODULE_PATH = 'test.unit.serialization.test_input'


class TestEnum(Enum):
    test_val = 'test_val'


def foo():
    pass


class Bar:
    def foo(self):
        pass


class Baz(BasicSerializer):
    def __init__(self, data_dct: dict = None):
        if data_dct is None:
            self.test_a = 'test_a'
            self.test_b = 42
            self.test_c = [self.test_a, self.test_b]
            self.test_d = {
                self.test_a: self.test_b
            }
        else:
            self.test_a = data_dct['test_a']
            self.test_b = data_dct['test_b']
            self.test_c = data_dct['test_c']
            self.test_d = data_dct['test_d']

    def __eq__(self, other):
        return (
            self.test_a == other.test_a and
            self.test_b == other.test_b and
            self.test_c == other.test_c and
            self.test_d == other.test_d
        )


class MockOperation(OperationSerializer):
    def __init__(self):
        self.operations_repo = 'operations_repo'

    def __eq__(self, other):
        return self.operations_repo == other.operations_repo


class MockNode(GraphNodeSerializer):
    def __init__(self, name: str, nodes_from: list = None):
        self.name = name
        self.nodes_from = nodes_from if nodes_from else []
        self._operator = '_operator'

    def __eq__(self, other):
        return (
            self.name == other.name and
            self.nodes_from == other.nodes_from and
            self._operator == other._operator
        )


MOCK_NODE_1 = MockNode('node1')
MOCK_NODE_2 = MockNode('node2')
MOCK_NODE_3 = MockNode('node3')
MOCK_NODE_1.nodes_from.extend([MOCK_NODE_2, MOCK_NODE_3])
MOCK_NODE_2.nodes_from.extend([MOCK_NODE_3])


class MockGraph(GraphSerializer):
    def __init__(self, nodes: list = None):
        self.nodes = nodes if nodes else []
        self.operator = 'operator'

    def __eq__(self, other):
        return (
            self.operator == other.operator and
            self.nodes == other.nodes
        )
