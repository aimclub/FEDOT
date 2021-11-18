from dataclasses import dataclass
from typing import Any, Dict, TypeVar

ClassOrFuncObject = TypeVar('ClassOrFuncObject')


@dataclass
class DumpObjectTestCase:
    test_input: ClassOrFuncObject
    test_answer: Dict[str, str]


@dataclass
class EncoderTestCase:
    test_input: ClassOrFuncObject
    test_answer: Dict[str, Any]


@dataclass
class GetClassCase:
    test_input: str
    test_answer: ClassOrFuncObject


@dataclass
class DecoderTestCase:
    test_input: Dict[str, Any]
    test_answer_type: ClassOrFuncObject
    test_answer: ClassOrFuncObject
