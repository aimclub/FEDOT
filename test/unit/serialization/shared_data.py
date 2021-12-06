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


class TestSerializableClass:
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
