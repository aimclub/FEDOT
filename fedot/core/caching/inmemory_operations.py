import torch
from pathlib import Path
from typing import Any, Callable
import pickle
import os
import uuid
import logging

from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.caching.responses import SaverResponse
from fedot.core.caching.normalization import (
    build_tensor_data_payload,
    build_preprocessing_model_payload)
from fedot.core.utils import CACHE_DIR
from fedot.core.data.tensor_data import TensorData


logger = logging.getLogger(__name__)


def _atomic_save(
    final_path: Path,
    writer: Callable[[Path], None],
    *,
    attempts: int = 2,
) -> bool:
    """
    Safely write file to final_path.

    Returns:
        True  - file was written by this call
        False - file already existed
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO @romankuklo: is it necessary to check if the file exists or index db is enough?
    if final_path.exists():
        logger.warning(f"File already exists: {final_path}")
        return True

    last_error = None

    for _ in range(attempts):
        tmp_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )

        try:
            writer(tmp_path)

            # If another process saved the same hash while we were writing,
            # do not overwrite it. Just remove our temp file.
            if final_path.exists():
                tmp_path.unlink(missing_ok=True)
                logger.warning(f"Failed to save cache file: {final_path} because of parallel write")
                return False

            os.replace(tmp_path, final_path)
            logger.info(f"Saved cache file: {final_path}")
            return True

        except Exception as ex:
            last_error = ex
            tmp_path.unlink(missing_ok=True)

    logger.error(f"Failed to save cache file: {final_path}", exc_info=last_error)
    return False


def save_tensor_data(data: TensorData, key: str) -> SaverResponse:
    ensure_cache_dirs()

    final_path = CACHE_DIR / "tensor_data" / f"{key}.pt"

    def writer(tmp_path: Path) -> None:
        payload = build_tensor_data_payload(data)
        torch.save(payload, tmp_path)

    written = _atomic_save(final_path, writer)

    return SaverResponse(
        key=key,
        kind="tensor_data",
        path=final_path,
        success=written,
    )


def save_preprocessing_model(data: Any, key: str) -> SaverResponse:
    ensure_cache_dirs()

    final_path = CACHE_DIR / "preprocessing_models" / f"{key}.pkl"

    def writer(tmp_path: Path) -> None:
        payload = build_preprocessing_model_payload(data)
        with open(tmp_path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    written = _atomic_save(final_path, writer)

    return SaverResponse(
        key=key,
        kind="preprocessing_model",
        path=final_path,
        success=written,
    )
