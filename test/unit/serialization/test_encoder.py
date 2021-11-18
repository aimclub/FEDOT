from copy import deepcopy
from uuid import UUID

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, OBJECT_ENCODING_KEY, encoder

from .dataclasses.serialization_dataclasses import EncoderTestCase
from .mocks.serialization_mocks import MockGraph, MockOperation, MockPipelineTemplate
from .shared_data import (
    MOCK_NODE_1,
    MOCK_NODE_2,
    MOCK_NODE_3,
    TEST_UUID,
    TestClass,
    TestEnum,
    TestSerializableClass,
    test_func
)

ENCODER_CASES = [
    EncoderTestCase(
        test_input=TestClass(),
        test_answer=TypeError()
    ),
    EncoderTestCase(
        test_input=UUID(TEST_UUID),
        test_answer={
            OBJECT_ENCODING_KEY: {'hex': TEST_UUID}
        }
    ),
    EncoderTestCase(
        test_input=TestEnum.test_val,
        test_answer={
            OBJECT_ENCODING_KEY: 'test_val'
        }
    ),
    EncoderTestCase(
        test_input=test_func,
        test_answer={}
    ),
    EncoderTestCase(
        test_input=TestClass().test_func,
        test_answer={}
    ),
    EncoderTestCase(
        test_input=TestSerializableClass(),
        test_answer={
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            }
        }
    ),
    EncoderTestCase(
        test_input=MockOperation(),
        test_answer={}
    ),
    EncoderTestCase(
        test_input=MockPipelineTemplate(),
        test_answer={}
    ),
    EncoderTestCase(
        test_input=MOCK_NODE_1,
        test_answer={
            'name': 'node1',
            'nodes_from': [
                MOCK_NODE_2,
                MOCK_NODE_3
            ]
        }
    ),
]

MOCK_NODE_1_COPY = deepcopy(MOCK_NODE_1)
MOCK_NODE_2_COPY = deepcopy(MOCK_NODE_2)
MOCK_NODE_3_COPY = deepcopy(MOCK_NODE_3)

ENCODER_CASES.extend([
    EncoderTestCase(
        test_input=MockGraph([MOCK_NODE_1_COPY, MOCK_NODE_2_COPY, MOCK_NODE_3_COPY]),
        test_answer={
            'nodes': [
                MOCK_NODE_1_COPY,
                MOCK_NODE_2_COPY,
                MOCK_NODE_3_COPY
            ]
        }
    ),
])


@pytest.mark.parametrize('case', ENCODER_CASES)
def test_encoder(case: EncoderTestCase):
    if isinstance(case.test_answer, Exception):
        with pytest.raises(type(case.test_answer)):
            encoder(case.test_input)
    else:
        if getattr(case.test_input, '__dict__', None) is not None:
            keys_before = vars(case.test_input).keys()
            encoded = {k: v for k, v in encoder(case.test_input).items() if k != CLASS_PATH_KEY}
            keys_after = vars(case.test_input).keys()
        else:
            keys_before = keys_after = {}
            encoded = {k: v for k, v in encoder(case.test_input).items() if k != CLASS_PATH_KEY}
        assert encoded == case.test_answer, 'Encoded json objects are not the same'
        assert keys_before == keys_after, 'Object instance was changed'
        if isinstance(case.test_input, MockGraph):
            assert vars(MOCK_NODE_1).keys() != vars(MOCK_NODE_1_COPY).keys()
            for node in case.test_input.nodes:
                assert getattr(node, '_serialization_id', None) is not None
