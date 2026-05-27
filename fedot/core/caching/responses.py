import os
from dataclasses import dataclass
from pathlib import Path


from fedot.core.utils import default_fedot_data_dir
from dotenv import load_dotenv


load_dotenv()

CACHE_PATH = default_fedot_data_dir() / Path(os.getenv("CACHE_PATH"))


@dataclass
class SaverResponse:
    key: str
    kind: str
    path: Path
    success: bool = False
