from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
