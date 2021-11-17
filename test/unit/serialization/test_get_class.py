from uuid import UUID

import pytest
from fedot.serializers.json_helpers import DELIMITER, _get_class

from .dataclasses.serialization_dataclasses import GetClassCase
from .test_input import *

GET_CLASS_CASES = [
    GetClassCase(
        input_data=f'{UUID.__module__}{DELIMITER}{UUID.__qualname__}',
        result=UUID
    ),
    GetClassCase(
        input_data=f'{TestEnum.__module__}{DELIMITER}{TestEnum.__qualname__}',
        result=TestEnum
    ),
    GetClassCase(
        input_data=f'{foo.__module__}{DELIMITER}{foo.__qualname__}',
        result=foo
    ),
    GetClassCase(
        input_data=f'{Bar().foo.__module__}{DELIMITER}{Bar().foo.__qualname__}',
        result=Bar.foo
    ),
    GetClassCase(
        input_data=f'{Baz.__module__}{DELIMITER}{Baz.__qualname__}',
        result=Baz
    ),
    GetClassCase(
        input_data=f'{MockOperation.__module__}{DELIMITER}{MockOperation.__qualname__}',
        result=MockOperation
    ),
    GetClassCase(
        input_data=f'{MockNode.__module__}{DELIMITER}{MockNode.__qualname__}',
        result=MockNode
    ),
    GetClassCase(
        input_data=f'{MockGraph.__module__}{DELIMITER}{MockGraph.__qualname__}',
        result=MockGraph
    )
]


@pytest.mark.parametrize('case', GET_CLASS_CASES)
def test_encoder(case: GetClassCase):
    cls_obj = _get_class(case.input_data)
    assert cls_obj == case.result, 'Decoded class is wrong'
