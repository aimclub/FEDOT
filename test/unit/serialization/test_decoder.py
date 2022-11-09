from uuid import UUID

import pytest

from fedot.core.serializers import CLASS_PATH_KEY, Serializer
from .dataclasses.serialization_dataclasses import DecoderTestCase
from .mocks.serialization_mocks import MockGraph, MockNode, MockOperation
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

DECODER_CASES = [
    DecoderTestCase(
        test_input={'a': 1, 'b': 2},
        test_answer={'a': 1, 'b': 2}
    ),
    DecoderTestCase(
        test_input={
            'hex': TEST_UUID,
            CLASS_PATH_KEY: UUID
        },
        test_answer=UUID(TEST_UUID)
    ),
    DecoderTestCase(
        test_input={
            CLASS_PATH_KEY: UUID
        },
        test_answer=KeyError()
    ),
    DecoderTestCase(
        test_input={
            'value': 'test_val',
            CLASS_PATH_KEY: TestEnum
        },
        test_answer=TestEnum.test_val
    ),
    DecoderTestCase(
        test_input={
            CLASS_PATH_KEY: test_func
        },
        test_answer=test_func
    ),
    DecoderTestCase(
        test_input={
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            },
            CLASS_PATH_KEY: TestSerializableClass
        },
        test_answer=TestSerializableClass({
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            }
        })
    ),
    DecoderTestCase(
        test_input={
            'operation_type': 'my_operation',
            CLASS_PATH_KEY: MockOperation
        },
        test_answer=MockOperation('my_operation')
    )
]

METHOD_FUNC = TestClass().test_func
DECODER_CASES.extend([
    DecoderTestCase(
        test_input={
            CLASS_PATH_KEY: METHOD_FUNC
        },
        test_answer=METHOD_FUNC
    )
])

DECODER_CASES.extend([
    DecoderTestCase(
        test_input={
            'name': 'node1',
            '_nodes_from': [
                MOCK_NODE_2,
                MOCK_NODE_3
            ],
            CLASS_PATH_KEY: MockNode
        },
        test_answer=MOCK_NODE_1
    ),
])

MOCK_NODE_1_SERIALIZED = MockNode(MOCK_NODE_1.name, [node.uid for node in MOCK_NODE_1.nodes_from])
MOCK_NODE_2_SERIALIZED = MockNode(MOCK_NODE_2.name, [node.uid for node in MOCK_NODE_2.nodes_from])
MOCK_NODE_3_SERIALIZED = MockNode(MOCK_NODE_3.name, [node.uid for node in MOCK_NODE_3.nodes_from])

DECODER_CASES.extend([
    DecoderTestCase(
        test_input={
            'nodes': [
                MOCK_NODE_1_SERIALIZED,
                MOCK_NODE_2_SERIALIZED,
                MOCK_NODE_3_SERIALIZED
            ],
            CLASS_PATH_KEY: MockGraph
        },
        test_answer=MockGraph([MOCK_NODE_1, MOCK_NODE_2, MOCK_NODE_3])
    )
])


@pytest.mark.parametrize('case', DECODER_CASES)
def test_decoder(case: DecoderTestCase, get_class_fixture, mock_classes_fixture):
    deserializer = Serializer()
    if isinstance(case.test_answer, Exception):
        with pytest.raises(type(case.test_answer)):
            deserializer.object_hook(case.test_input)
    else:
        decoded = deserializer.object_hook(case.test_input)
        assert isinstance(decoded, type(case.test_answer)), 'Decoded object has wrong type'
        assert decoded == case.test_answer, f'Object was decoded incorrectly'
