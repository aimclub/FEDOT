from dataclasses import dataclass
from typing import Any, Dict, TypeVar

ClassOrFuncObject = TypeVar('ClassOrFuncObject')


@dataclass
class DumpObjectTestCase:
    input_data: ClassOrFuncObject
    result: Dict[str, str]


@dataclass
class EncoderTestCase:
    input_data: ClassOrFuncObject
    result: Dict[str, Any]


@dataclass
class GetClassCase:
    input_data: str
    result: ClassOrFuncObject


@dataclass
class DecoderTestCase:
    input_data: Dict[str, Any]
    result_type: ClassOrFuncObject
    result: ClassOrFuncObject
