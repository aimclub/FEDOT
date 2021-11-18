from copy import deepcopy
from test.unit.serialization.test_encoder import MOCK_NODE_1_COPY
from types import FunctionType, MethodType
from uuid import UUID

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, OBJECT_ENCODING_KEY, decoder

from .dataclasses.serialization_dataclasses import DecoderTestCase
from .fixtures.serialization_fixtures import _get_class_fixture
from .test_input import (
    MOCK_NODE_1,
    MOCK_NODE_2,
    MOCK_NODE_3,
    TEST_UUID,
    Bar,
    Baz,
    MockGraph,
    MockNode,
    MockOperation,
    MockPipelineTemplate,
    TestEnum,
    foo
)

DECODER_CASES = [
    DecoderTestCase(
        input_data={'a': 1, 'b': 2},
        result_type=dict,
        result={'a': 1, 'b': 2}
    ),
    DecoderTestCase(
        input_data={
            OBJECT_ENCODING_KEY: {'hex': TEST_UUID},
            CLASS_PATH_KEY: UUID
        },
        result_type=UUID,
        result=UUID(TEST_UUID)
    ),
    DecoderTestCase(
        input_data={
            CLASS_PATH_KEY: UUID
        },
        result_type=TypeError,
        result=TypeError()
    ),
    DecoderTestCase(
        input_data={
            OBJECT_ENCODING_KEY: 'test_val',
            CLASS_PATH_KEY: TestEnum
        },
        result_type=TestEnum,
        result=TestEnum.test_val
    ),
    DecoderTestCase(
        input_data={
            CLASS_PATH_KEY: foo
        },
        result_type=FunctionType,
        result=foo
    ),
    DecoderTestCase(
        input_data={
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            },
            CLASS_PATH_KEY: Baz
        },
        result_type=Baz,
        result=Baz({
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            }
        })
    ),
    DecoderTestCase(
        input_data={
            CLASS_PATH_KEY: MockOperation
        },
        result_type=MockOperation,
        result=MockOperation()
    ),
    DecoderTestCase(
        input_data={
            CLASS_PATH_KEY: MockPipelineTemplate
        },
        result_type=MockPipelineTemplate,
        result=MockPipelineTemplate()
    )
]

METHOD_FUNC = Bar().foo
DECODER_CASES.extend([
    DecoderTestCase(
        input_data={
            CLASS_PATH_KEY: METHOD_FUNC
        },
        result_type=MethodType,
        result=METHOD_FUNC
    )
])

MOCK_NODE_1_COPY = deepcopy(MOCK_NODE_1)
MOCK_NODE_2_COPY = deepcopy(MOCK_NODE_2)
MOCK_NODE_3_COPY = deepcopy(MOCK_NODE_3)

DECODER_CASES.extend([
    DecoderTestCase(
        input_data={
            'name': 'node1',
            'nodes_from': [
                MOCK_NODE_2_COPY,
                MOCK_NODE_3_COPY
            ],
            '_serialization_id': 0,
            CLASS_PATH_KEY: MockNode
        },
        result_type=MockNode,
        result=MOCK_NODE_1_COPY
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
        input_data={
            'nodes': [
                MOCK_NODE_1_COPY,
                MOCK_NODE_2_COPY,
                MOCK_NODE_3_COPY
            ],
            CLASS_PATH_KEY: MockGraph
        },
        result_type=MockGraph,
        result=MockGraph([MOCK_NODE_1_COPY, MOCK_NODE_2_COPY, MOCK_NODE_3_COPY])
    )
])


@pytest.mark.parametrize('case', DECODER_CASES)
def test_decoder(case: DecoderTestCase, _get_class_fixture):
    if isinstance(case.result, Exception):
        with pytest.raises(type(case.result)):
            decoder(case.input_data)
    else:
        decoded = decoder(case.input_data)
        assert type(decoded) == case.result_type, 'Decoded object has wrong type'
        assert decoded == case.result, f'Object was decoded incorrectly'
