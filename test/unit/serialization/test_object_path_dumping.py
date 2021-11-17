from uuid import UUID, uuid4

import pytest
from fedot.serializers.json_helpers import CLASS_PATH_KEY, DELIMITER, dump_path_to_obj

from .dataclasses.serialization_dataclasses import DumpObjectTestCase
from .test_input import *

DUMPING_CASES = [
    DumpObjectTestCase(
        input_data=UUID(TEST_UUID),
        result={
            CLASS_PATH_KEY: f'uuid{DELIMITER}UUID'
        }
    ),
    DumpObjectTestCase(
        input_data=TestEnum.test_val,
        result={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}TestEnum'
        }
    ),
    DumpObjectTestCase(
        input_data=foo,
        result={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}foo'
        }
    ),
    DumpObjectTestCase(
        input_data=Bar().foo,
        result={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}Bar.foo'
        }
    ),
    DumpObjectTestCase(
        input_data=Baz(),
        result={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}Baz'
        }
    )
]


@pytest.mark.parametrize('case', DUMPING_CASES)
def test_dumping(case: DumpObjectTestCase):
    dumped = dump_path_to_obj(case.input_data)
    assert dumped == case.result, f'Object dumping works incorrectly!'
