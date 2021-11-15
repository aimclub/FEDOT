from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union

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
class DecoderTestCase:
    input_data: Dict[str, Any]
    result: ClassOrFuncObject
