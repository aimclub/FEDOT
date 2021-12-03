from enum import Enum

from .mocks.serialization_mocks import MockNode

TEST_UUID = '41d79d06c3d8478f89e7d1008c96a864'
TEST_MODULE_PATH = 'test.unit.serialization.shared_data'
MOCK_NODE_1 = MockNode('node1')
MOCK_NODE_2 = MockNode('node2')
MOCK_NODE_3 = MockNode('node3')
MOCK_NODE_1.nodes_from.extend([MOCK_NODE_2, MOCK_NODE_3])
MOCK_NODE_2.nodes_from.extend([MOCK_NODE_3])


class TestEnum(Enum):
    test_val = 'test_val'


def test_func():
    pass


class TestClass:
    def test_func(self):
        pass
