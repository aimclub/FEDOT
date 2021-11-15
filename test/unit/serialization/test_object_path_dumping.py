import uuid
from enum import Enum
from uuid import uuid4

import pytest
from fedot.shared.serializers.basic_serializer import BasicSerializer
from fedot.shared.serializers.json_helpers import CLASS_PATH_KEY, DELIMITER, dump_path_to_obj

from .dataclasses.serialization_dataclasses import DumpObjectTestCase

test_uuid = uuid4()


class TestEnum(Enum):
    test_val = 'test_val'


def foo():
    pass


class Bar:
    def foo():
        pass


class Baz(BasicSerializer):
    pass


cur_module_path = 'test.unit.serialization.test_object_path_dumping'

DUMPING_CASES = [
    DumpObjectTestCase(
        input_data=test_uuid,
        result={
            CLASS_PATH_KEY: f'uuid{DELIMITER}UUID'
        }
    ),
    DumpObjectTestCase(
        input_data=TestEnum.test_val,
        result={
            CLASS_PATH_KEY: f'{cur_module_path}{DELIMITER}TestEnum'
        }
    ),
    DumpObjectTestCase(
        input_data=foo,
        result={
            CLASS_PATH_KEY: f'{cur_module_path}{DELIMITER}foo'
        }
    ),
    DumpObjectTestCase(
        input_data=Bar().foo,
        result={
            CLASS_PATH_KEY: f'{cur_module_path}{DELIMITER}Bar.foo'
        }
    ),
    DumpObjectTestCase(
        input_data=Baz(),
        result={
            CLASS_PATH_KEY: f'{cur_module_path}{DELIMITER}Baz'
        }
    )
]


@pytest.mark.parametrize('case', DUMPING_CASES)
def test_dumping(case: DumpObjectTestCase):
    dumped = dump_path_to_obj(case.input_data)
    assert dumped == case.result, f'Object dumping works incorrectly!'
