from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from fedot.core.caching.enums import CacheModeEnum


@dataclass
class SaverResponse:
    key: str
    kind: str
    path: Path
    success: bool = False


@dataclass
class DataCacherLoaderResponse:
    input_hash: str = None
    output_hash: str = None
    operation_hash: str = None
    model_hash: str = None
    path: Path = None
    success: bool = False
    data: Any = None
    model: Any = None


@dataclass
class NormalizedCleaningStrategyResponse:
    mode: CacheModeEnum
    tensor_data_hashes: Optional[List[str]]
    ratio_first_tensor_data: Optional[int]
