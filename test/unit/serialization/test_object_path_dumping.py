from uuid import UUID, uuid4

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, DELIMITER, dump_path_to_obj

from .dataclasses.serialization_dataclasses import DumpObjectTestCase
from .shared_data import TEST_INPUT_MODULE_PATH, TEST_UUID, Bar, Baz, TestEnum, foo

DUMPING_CASES = [
    DumpObjectTestCase(
        test_input=UUID(TEST_UUID),
        test_answer={
            CLASS_PATH_KEY: f'uuid{DELIMITER}UUID'
        }
    ),
    DumpObjectTestCase(
        test_input=TestEnum.test_val,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}TestEnum'
        }
    ),
    DumpObjectTestCase(
        test_input=foo,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}foo'
        }
    ),
    DumpObjectTestCase(
        test_input=Bar().foo,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}Bar.foo'
        }
    ),
    DumpObjectTestCase(
        test_input=Baz(),
        test_answer={
            CLASS_PATH_KEY: f'{TEST_INPUT_MODULE_PATH}{DELIMITER}Baz'
        }
    )
]


@pytest.mark.parametrize('case', DUMPING_CASES)
def test_dumping(case: DumpObjectTestCase):
    dumped = dump_path_to_obj(case.test_input)
    assert dumped == case.test_answer, f'Object dumping works incorrectly!'
