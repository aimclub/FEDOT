from dataclasses import dataclass
from typing import Any, Dict

from golem.serializers import INSTANCE_OR_CALLABLE


@dataclass
class DumpObjectTestCase:
    test_input: INSTANCE_OR_CALLABLE
    test_answer: Dict[str, str]


@dataclass
class EncoderTestCase:
    test_input: INSTANCE_OR_CALLABLE
    test_answer: Dict[str, Any]


@dataclass
class GetClassCase:
    test_input: str
    test_answer: INSTANCE_OR_CALLABLE


@dataclass
class DecoderTestCase:
    test_input: Dict[str, Any]
    test_answer: INSTANCE_OR_CALLABLE
