from copy import deepcopy
from uuid import UUID

import pytest
from fedot.core.serializers.json_helpers import CLASS_PATH_KEY, OBJECT_ENCODING_KEY, encoder

from .dataclasses.serialization_dataclasses import EncoderTestCase
from .test_input import (
    MOCK_NODE_1,
    MOCK_NODE_2,
    MOCK_NODE_3,
    TEST_UUID,
    Bar,
    Baz,
    MockGraph,
    MockOperation,
    MockPipelineTemplate,
    TestEnum,
    foo
)

ENCODER_CASES = [
    EncoderTestCase(
        input_data=Bar(),
        result=TypeError()
    ),
    EncoderTestCase(
        input_data=UUID(TEST_UUID),
        result={
            OBJECT_ENCODING_KEY: {'hex': TEST_UUID}
        }
    ),
    EncoderTestCase(
        input_data=TestEnum.test_val,
        result={
            OBJECT_ENCODING_KEY: 'test_val'
        }
    ),
    EncoderTestCase(
        input_data=foo,
        result={}
    ),
    EncoderTestCase(
        input_data=Bar().foo,
        result={}
    ),
    EncoderTestCase(
        input_data=Baz(),
        result={
            'test_a': 'test_a',
            'test_b': 42,
            'test_c': ['test_a', 42],
            'test_d': {
                'test_a': 42
            }
        }
    ),
    EncoderTestCase(
        input_data=MockOperation(),
        result={}
    ),
    EncoderTestCase(
        input_data=MockPipelineTemplate(),
        result={}
    ),
    EncoderTestCase(
        input_data=MOCK_NODE_1,
        result={
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
        input_data=MockGraph([MOCK_NODE_1_COPY, MOCK_NODE_2_COPY, MOCK_NODE_3_COPY]),
        result={
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
    if isinstance(case.result, Exception):
        with pytest.raises(type(case.result)):
            encoder(case.input_data)
    else:
        if getattr(case.input_data, '__dict__', None) is not None:
            keys_before = vars(case.input_data).keys()
            encoded = {k: v for k, v in encoder(case.input_data).items() if k != CLASS_PATH_KEY}
            keys_after = vars(case.input_data).keys()
        else:
            keys_before = keys_after = {}
            encoded = {k: v for k, v in encoder(case.input_data).items() if k != CLASS_PATH_KEY}
        assert encoded == case.result, 'Encoded json objects are not the same'
        assert keys_before == keys_after, 'Object instance was changed'
        if isinstance(case.input_data, MockGraph):
            assert vars(MOCK_NODE_1).keys() != vars(MOCK_NODE_1_COPY).keys()
            for node in case.input_data.nodes:
                assert getattr(node, '_serialization_id', None) is not None
