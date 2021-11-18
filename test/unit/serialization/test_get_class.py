from uuid import UUID

import pytest
from fedot.core.serializers.json_helpers import DELIMITER, _get_class

from .dataclasses.serialization_dataclasses import GetClassCase
from .mocks.serialization_mocks import MockGraph, MockNode, MockOperation
from .shared_data import Bar, Baz, TestEnum, foo

GET_CLASS_CASES = [
    GetClassCase(
        test_input=f'{UUID.__module__}{DELIMITER}{UUID.__qualname__}',
        test_answer=UUID
    ),
    GetClassCase(
        test_input=f'{TestEnum.__module__}{DELIMITER}{TestEnum.__qualname__}',
        test_answer=TestEnum
    ),
    GetClassCase(
        test_input=f'{foo.__module__}{DELIMITER}{foo.__qualname__}',
        test_answer=foo
    ),
    GetClassCase(
        test_input=f'{Bar().foo.__module__}{DELIMITER}{Bar().foo.__qualname__}',
        test_answer=Bar.foo
    ),
    GetClassCase(
        test_input=f'{Baz.__module__}{DELIMITER}{Baz.__qualname__}',
        test_answer=Baz
    ),
    GetClassCase(
        test_input=f'{MockOperation.__module__}{DELIMITER}{MockOperation.__qualname__}',
        test_answer=MockOperation
    ),
    GetClassCase(
        test_input=f'{MockNode.__module__}{DELIMITER}{MockNode.__qualname__}',
        test_answer=MockNode
    ),
    GetClassCase(
        test_input=f'{MockGraph.__module__}{DELIMITER}{MockGraph.__qualname__}',
        test_answer=MockGraph
    )
]


@pytest.mark.parametrize('case', GET_CLASS_CASES)
def test_encoder(case: GetClassCase):
    cls_obj = _get_class(case.test_input)
    assert cls_obj == case.test_answer, 'Decoded class is wrong'
