from uuid import UUID

import pytest
from golem.serializers import CLASS_PATH_KEY, MODULE_X_NAME_DELIMITER, Serializer

from .dataclasses.serialization_dataclasses import DumpObjectTestCase
from .shared_data import TEST_MODULE_PATH, TEST_UUID, TestClass, TestEnum, TestSerializableClass, test_func

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
    ),
    DumpObjectTestCase(
        test_input=TestSerializableClass(),
        test_answer={
            CLASS_PATH_KEY: f'{TEST_MODULE_PATH}{MODULE_X_NAME_DELIMITER}TestSerializableClass'
        }
    )
]


@pytest.mark.parametrize('case', DUMPING_CASES)
def test_dumping(case: DumpObjectTestCase):
    dumped = Serializer.dump_path_to_obj(case.test_input)
    assert dumped == case.test_answer, f'Object dumping works incorrectly!'
