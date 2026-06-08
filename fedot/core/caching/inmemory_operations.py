import torch
import cloudpickle
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
    build_preprocessing_model_payload,
    prepare_loaded_preprocessing_model,
    restore_tensor_data_payload,
)
from fedot.core.utils import CACHE_DIR
from fedot.core.data.tensor_data import TensorData
from fedot.preprocessing.planner import PreprocessingPlan


logger = logging.getLogger(__name__)


def _atomic_save(
    final_path: Path,
    writer: Callable[[Path], None],
    *,
    attempts: int = 2,
) -> bool:
    """
    Safely write a cache artifact using a temporary file and ``os.replace``.

    Args:
        final_path: Destination path under ``CACHE_DIR``.
        writer: Callable that writes content to a temporary path.
        attempts: Number of write attempts before giving up.

    Returns:
        ``True`` when this call created ``final_path`` or it already existed.
        ``False`` when a concurrent writer won the race or all attempts failed.
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
    """
    Serialize ``TensorData`` to ``CACHE_DIR/tensor_data/{key}.pt``.

    Args:
        data: Prepared tensor container to persist.
        key: Content hash used as the file stem.

    Returns:
        ``SaverResponse`` with the final path and write status.
    """
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


def save_preprocessing_plan(data: PreprocessingPlan, key: str) -> SaverResponse:
    """
    Pickle a preprocessing plan to ``CACHE_DIR/preprocessing_plans/{key}.pkl``.

    Args:
        data: Preprocessing plan instance.
        key: Content hash used as the file stem.

    Returns:
        ``SaverResponse`` with the final path and write status.
    """
    ensure_cache_dirs()

    final_path = CACHE_DIR / "preprocessing_plans" / f"{key}.pkl"

    def writer(tmp_path: Path) -> None:
        with open(tmp_path, "wb") as file:
            cloudpickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    written = _atomic_save(final_path, writer)

    return SaverResponse(
        key=key,
        kind="preprocessing_plan",
        path=final_path,
        success=written,
    )


def save_preprocessing_model(data: Any, key: str) -> SaverResponse:
    """
    Pickle a fitted preprocessing model to ``CACHE_DIR/preprocessing_models/{key}.pkl``.

    The model state is normalized before pickling so torch tensors and nested
    dataclasses are cache-safe.

    Args:
        data: Fitted preprocessing handler instance.
        key: Content hash used as the file stem.

    Returns:
        ``SaverResponse`` with the final path and write status.
    """
    ensure_cache_dirs()

    final_path = CACHE_DIR / "preprocessing_models" / f"{key}.pkl"

    def writer(tmp_path: Path) -> None:
        payload = build_preprocessing_model_payload(data)
        with open(tmp_path, "wb") as file:
            cloudpickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    written = _atomic_save(final_path, writer)

    return SaverResponse(
        key=key,
        kind="preprocessing_model",
        path=final_path,
        success=written,
    )


def load_pt_file(source: str, hash: str = None, kind: str = None) -> Any:
    """
    Load cached ``TensorData`` from a ``.pt`` file.

    Args:
        source: Path to the torch cache file.
        hash: Expected fingerprint validated after restore.
        kind: Optional artifact kind; only ``tensor_data`` is supported.

    Returns:
        Restored ``TensorData`` on the active backend device.

    Raises:
        ValueError: For unsupported ``kind`` or hash mismatch.
    """
    if kind not in (None, "tensor_data"):
        raise ValueError(f"Unsupported .pt cache kind: {kind}")

    payload = _torch_load(source)
    data = restore_tensor_data_payload(payload)
    _validate_loaded_hash(data, hash)
    logger.info(f"Loaded cached tensor data from {source}")
    return data


def load_pkl_file(source: str, hash: str = None, kind: str = None) -> Any:
    """
    Load a pickled preprocessing artifact from a ``.pkl`` file.

    Args:
        source: Path to the pickle cache file.
        hash: Expected fingerprint validated after loading.
        kind: Artifact kind (`preprocessing_model` or `preprocessing_plan`).

    Returns:
        Restored Python object, with tensors moved to the active backend for models.

    Raises:
        ValueError: For unsupported ``kind`` or hash mismatch.
    """
    if kind not in (None, "preprocessing_model", "preprocessing_plan"):
        raise ValueError(f"Unsupported .pkl cache kind: {kind}")

    with open(source, "rb") as file:
        data = pickle.load(file)

    if kind in (None, "preprocessing_model"):
        data = prepare_loaded_preprocessing_model(data)
    _validate_loaded_hash(data, hash)
    logger.info(f"Loaded cached {kind or 'preprocessing_model'} from {source}")
    return data


def _torch_load(source: str) -> Any:
    """Load a torch-serialized payload from disk on CPU."""
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _validate_loaded_hash(data: Any, expected_hash: str = None) -> None:
    """
    Verify that a loaded cache object matches the expected content hash.

    Args:
        data: Object restored from disk.
        expected_hash: Expected ``Hasher.hash`` value. Skipped when ``None``.

    Raises:
        ValueError: When the recomputed hash differs from ``expected_hash``.
    """
    if expected_hash is None:
        return

    from fedot.core.caching.hasher import Hasher

    actual_hash = Hasher.hash(data)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Loaded cache hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
