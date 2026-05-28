from dataclasses import dataclass
from pathlib import Path


@dataclass
class SaverResponse:
    key: str
    kind: str
    path: Path
    success: bool = False
