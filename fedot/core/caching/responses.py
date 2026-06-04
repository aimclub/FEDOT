from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from fedot.core.caching.enums import CacheModeEnum


@dataclass
class SaverResponse:
    """
    Result of a cache write performed by ``Saver`` or ``inmemory_operations``.

    Attributes:
        key: Content hash used as the artifact identifier.
        kind: Artifact category (`tensor_data`, `preprocessing_model`, etc.).
        path: Final on-disk path of the cached file.
        success: Whether this call created the artifact file.
    """
    key: str
    kind: str
    path: Path
    success: bool = False


@dataclass
class DataCacherLoaderResponse:
    """
    Result of a cache lookup performed by ``Cacher.load_*`` methods.

    Attributes:
        input_hash: Fingerprint of the preprocessing input.
        output_hash: Fingerprint of cached tensor output, when applicable.
        operation_hash: Fingerprint of the preprocessing plan or operation.
        model_hash: Fingerprint of a cached preprocessing model.
        path: Resolved artifact path from the index.
        success: Whether the requested object was loaded.
        data: Loaded ``TensorData`` when ``success`` is ``True``.
        model: Loaded preprocessing model when ``success`` is ``True``.
    """
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
    """
    Normalized parameters produced by ``normilize_cleaning_strategy``.

    Attributes:
        mode: Resolved cleaning mode.
        tensor_data_hashes: Hashes to delete for ``TENSOR_DATA`` mode.
        ratio_first_tensor_data: Number of oldest tensor files to delete for
            ``FIRST_N_TENSOR_DATA`` mode.
    """
    mode: CacheModeEnum
    tensor_data_hashes: Optional[List[str]]
    ratio_first_tensor_data: Optional[int]
