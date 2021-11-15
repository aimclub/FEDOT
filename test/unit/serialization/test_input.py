from enum import Enum

from fedot.shared import BasicSerializer

TEST_UUID = '41d79d06c3d8478f89e7d1008c96a864'
TEST_INPUT_MODULE_PATH = 'test.unit.serialization.test_input'


class TestEnum(Enum):
    test_val = 'test_val'


def foo():
    pass


class Bar:
    def foo():
        pass


class Baz(BasicSerializer):
    pass
