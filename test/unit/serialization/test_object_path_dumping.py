from uuid import UUID, uuid4

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, MODULE_X_NAME_DELIMITER, dump_path_to_obj

from .dataclasses.serialization_dataclasses import DumpObjectTestCase
from .shared_data import TEST_MODULE_PATH, TEST_UUID, TestClass, TestEnum, test_func

DUMPING_CASES = [
    DumpObjectTestCase(
        test_input=UUID(TEST_UUID),
        test_answer={
            CLASS_PATH_KEY: f'uuid{MODULE_X_NAME_DELIMITER}UUID'
        }
    ),
    DumpObjectTestCase(
        test_input=TestEnum.test_val,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_MODULE_PATH}{MODULE_X_NAME_DELIMITER}TestEnum'
        }
    ),
    DumpObjectTestCase(
        test_input=test_func,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_MODULE_PATH}{MODULE_X_NAME_DELIMITER}test_func'
        }
    ),
    DumpObjectTestCase(
        test_input=TestClass().test_func,
        test_answer={
            CLASS_PATH_KEY: f'{TEST_MODULE_PATH}{MODULE_X_NAME_DELIMITER}TestClass.test_func'
        }
    )
]


@pytest.mark.parametrize('case', DUMPING_CASES)
def test_dumping(case: DumpObjectTestCase):
    dumped = dump_path_to_obj(case.test_input)
    assert dumped == case.test_answer, f'Object dumping works incorrectly!'
