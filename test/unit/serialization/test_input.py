from enum import Enum
from uuid import UUID, uuid4

from fedot.shared import BasicSerializer

TEST_UUID = uuid4()
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
