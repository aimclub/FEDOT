from copy import deepcopy
from test.unit.serialization.test_encoder import MOCK_NODE_1_COPY
from types import FunctionType, MethodType
from uuid import UUID

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, OBJECT_ENCODING_KEY, decoder

from .dataclasses.serialization_dataclasses import DecoderTestCase
from .fixtures.serialization_fixtures import _get_class_fixture
from .mocks.serialization_mocks import MockGraph, MockNode, MockOperation, MockPipelineTemplate
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
            OBJECT_ENCODING_KEY: {'hex': TEST_UUID},
            CLASS_PATH_KEY: UUID
        },
        test_answer=UUID(TEST_UUID)
    ),
    DecoderTestCase(
        test_input={
            CLASS_PATH_KEY: UUID
        },
        test_answer=TypeError()
    ),
    DecoderTestCase(
        test_input={
            OBJECT_ENCODING_KEY: 'test_val',
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
            CLASS_PATH_KEY: MockOperation
        },
        test_answer=MockOperation()
    ),
    DecoderTestCase(
        test_input={
            CLASS_PATH_KEY: MockPipelineTemplate
        },
        test_answer=MockPipelineTemplate()
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

MOCK_NODE_1_COPY = deepcopy(MOCK_NODE_1)
MOCK_NODE_2_COPY = deepcopy(MOCK_NODE_2)
MOCK_NODE_3_COPY = deepcopy(MOCK_NODE_3)

DECODER_CASES.extend([
    DecoderTestCase(
        test_input={
            'name': 'node1',
            'nodes_from': [
                MOCK_NODE_2_COPY,
                MOCK_NODE_3_COPY
            ],
            '_serialization_id': 0,
            CLASS_PATH_KEY: MockNode
        },
        test_answer=MOCK_NODE_1_COPY
    ),
])

MOCK_NODE_3_COPY = MockNode('name3')
MOCK_NODE_3_COPY._serialization_id = 2

MOCK_NODE_2_COPY = MockNode('name2')
MOCK_NODE_2_COPY._serialization_id = 1
MOCK_NODE_3_COPY_1 = MockNode('name2')
MOCK_NODE_3_COPY_1._serialization_id = 2
vars(MOCK_NODE_2_COPY).update({
    'nodes_from': [
        MOCK_NODE_3_COPY_1
    ]
})

MOCK_NODE_1_COPY = MockNode('name1')
MOCK_NODE_1_COPY._serialization_id = 0
MOCK_NODE_2_COPY_2 = MockNode('name2')
MOCK_NODE_2_COPY_2._serialization_id = 1
MOCK_NODE_3_COPY_2 = MockNode('name3')
MOCK_NODE_3_COPY_2._serialization_id = 2
vars(MOCK_NODE_2_COPY_2).update({
    'nodes_from': [
        MOCK_NODE_3_COPY_2
    ]
})
MOCK_NODE_3_COPY_3 = MockNode('name3')
MOCK_NODE_3_COPY_3._serialization_id = 2
vars(MOCK_NODE_1_COPY).update({
    'nodes_from': [
        MOCK_NODE_2_COPY_2,
        MOCK_NODE_3_COPY_3
    ]
})

DECODER_CASES.extend([
    DecoderTestCase(
        test_input={
            'nodes': [
                MOCK_NODE_1_COPY,
                MOCK_NODE_2_COPY,
                MOCK_NODE_3_COPY
            ],
            CLASS_PATH_KEY: MockGraph
        },
        test_answer=MockGraph([MOCK_NODE_1_COPY, MOCK_NODE_2_COPY, MOCK_NODE_3_COPY])
    )
])


@pytest.mark.parametrize('case', DECODER_CASES)
def test_decoder(case: DecoderTestCase, _get_class_fixture):
    if isinstance(case.test_answer, Exception):
        with pytest.raises(type(case.test_answer)):
            decoder(case.test_input)
    else:
        decoded = decoder(case.test_input)
        assert isinstance(decoded, type(case.test_answer)), 'Decoded object has wrong type'
        assert decoded == case.test_answer, f'Object was decoded incorrectly'
