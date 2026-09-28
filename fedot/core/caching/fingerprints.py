import hashlib
import inspect
from typing import Any, Optional

import numpy as np
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data import TensorData
from fedot.core.caching.normalization import _stable_bytes, normalize_for_hash, stable_hash
from fedot.core.caching.sampling import sample_row_positions
from fedot.core.common.tools import to_numpy


def _blake_hash_bytes(data: bytes, digest_size: int = 16) -> str:
    """Return a Blake2b hash for already prepared bytes."""
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def _array_to_bytes(array: Any) -> bytes:
    """
    Convert an array fragment to stable bytes.

    Numeric arrays are hashed by raw contiguous bytes. Object arrays are routed
    through canonical JSON to support strings/categories/mixed values.
    """
    array = to_numpy(array)

    if array.dtype.hasobject:
        return _stable_bytes(array.tolist())

    return np.ascontiguousarray(array).tobytes()


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a tensor or tensor fragment to stable bytes.

    Sparse tensors are densified, and only the supplied tensor is moved to CPU.
    """
    tensor = tensor.detach()

    if tensor.is_sparse:
        tensor = tensor.to_dense()

    tensor = tensor.cpu().contiguous()

    return tensor.numpy().tobytes()


def _tensor_shape(x: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    """Return tensor shape as a tuple, preserving `None`."""
    if x is None:
        return None
    return tuple(int(dim) for dim in x.shape)


def _tensor_dtype(x: Optional[torch.Tensor]) -> Optional[str]:
    """Return tensor dtype as a string, preserving `None`."""
    if x is None:
        return None
    return str(x.dtype)


def tensor_features_for_hash(features: torch.Tensor) -> torch.Tensor:
    """
    Return a row-oriented tensor view used for `TensorData` feature sampling.

    Two-dimensional tensors are already interpreted as `(samples, features)`.
    Higher-dimensional tensors are flattened over all leading dimensions and
    keep the last dimension as features; for `(S, C, F)` this gives `(S*C, F)`.
    """
    if features.ndim <= 2:
        return features

    return features.reshape(-1, features.shape[-1])


def build_tensordata_metadata(td: TensorData) -> dict[str, Any]:
    """
    Build metadata that affects the meaning of `TensorData`.

    Target and prediction values are intentionally not sampled here; only their
    shapes and dtypes are included.
    """
    return {
        "task": td.task,
        "data_type": td.data_type,
        "state": td.state,
        "features_shape": _tensor_shape(td.features),
        "features_dtype": _tensor_dtype(td.features),
        "target_shape": _tensor_shape(td.target),
        "target_dtype": _tensor_dtype(td.target),
        "categorical_idx": td.categorical_idx,
        "numerical_idx": td.numerical_idx,
        "features_names": td.features_names,
        "idx_mapping": td.idx_mapping,
        "ts_orientation": td.ts_orientation,
        "ts_terms_idx": td.ts_terms_idx,
        "ts_forecast_horizon": td.ts_forecast_horizon,
        "ts_init_shape": td.ts_init_shape,
    }


def get_hash_raw_features(
    features: Any,
    row_positions: list[int],
    target: Any = None,
    digest_size: int = 16,
) -> str:
    """
    Hash raw feature array metadata and sampled rows.

    Args:
        features: NumPy or CuPy feature array.
        row_positions: Row indices selected from the first axis.
        target: Optional target array included in the fingerprint.
        digest_size: Blake2b digest size in bytes.

    Returns:
        Hexadecimal hash string.
    """
    metadata = {
        "features_shape": tuple(int(dim) for dim in features.shape),
        "features_dtype": str(features.dtype),
        "target_shape": None if target is None else tuple(int(dim) for dim in target.shape),
        "target_dtype": None if target is None else str(target.dtype),
    }

    h = hashlib.blake2b(digest_size=digest_size)
    h.update(_stable_bytes(metadata))

    if row_positions:
        h.update(_array_to_bytes(features[row_positions]))

    if target is not None:
        h.update(_array_to_bytes(target))

    return h.hexdigest()


def get_hash_tensordata(
    td: TensorData,
    min_rows: int = 8,
    max_rows: int = 64,
    digest_size: int = 16,
) -> str:
    """
    Hash `TensorData` metadata and sampled feature rows.

    Args:
        td: Prepared tensor data container.
        min_rows: Lower bound for the requested feature row sample size.
        max_rows: Upper bound for the requested feature row sample size.
        digest_size: Blake2b digest size in bytes.

    Returns:
        Hexadecimal hash string.
    """
    metadata = build_tensordata_metadata(td)

    h = hashlib.blake2b(digest_size=digest_size)
    h.update(_stable_bytes(metadata))

    features = tensor_features_for_hash(td.features)
    row_positions = sample_row_positions(
        shape=tuple(features.shape),
        dtype=features.dtype,
        min_rows=min_rows,
        max_rows=max_rows,
    )

    if row_positions:
        index = torch.as_tensor(row_positions, device=features.device, dtype=torch.long)
        sample = features.index_select(dim=0, index=index)
        h.update(_tensor_to_bytes(sample))

    target = td.target
    if target is not None:
        h.update(_tensor_to_bytes(target))

    return h.hexdigest()


def implementation_signature(implementation: Any) -> Optional[dict[str, Any]]:
    """Return a stable class identifier for a preprocessing implementation."""
    if implementation is None:
        return None

    cls = implementation if inspect.isclass(implementation) else implementation.__class__

    return {
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
    }


DEFAULT_EXCLUDED_MODEL_FIELDS = frozenset({
    "logger",          # logging handles are runtime-only and not model state
    "cache",           # caches are derived runtime data, not fitted parameters
    "cacher",          # cache helpers can include non-deterministic/runtime state
    "device_context",  # backend/device context should not affect fitted model hash
})


def get_model_attributes(
    model: Any,
    exclude_fields: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Extract fitted preprocessing model state from `__dict__`.

    The class path is always included so unfitted handlers/classes of different
    types do not collapse to the same hash.
    """
    exclude_fields = exclude_fields or DEFAULT_EXCLUDED_MODEL_FIELDS
    cls = model if inspect.isclass(model) else model.__class__

    model_state = {
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
        "attributes": {},
    }

    if inspect.isclass(model) or not hasattr(model, "__dict__"):
        return model_state

    attributes = {}
    for key, value in vars(model).items():
        if key in exclude_fields or key.startswith("__"):
            continue

        attributes[key] = normalize_for_hash(value)

    model_state["attributes"] = attributes
    return model_state


def preprocessing_step_hash(step: Any, digest_size: int = 16) -> str:
    """
    Hash a single preprocessing step definition and implementation state.

    The implementation state is included so fitted preprocessing handlers with
    different learned parameters produce different hashes.
    """
    step_dict = {
        "step": normalize_for_hash(step.step),
        "method": normalize_for_hash(step.method),
        "features_idx": normalize_for_hash(step.features_idx),
        "implementation": implementation_signature(step.implementation),
        "implementation_state": get_model_attributes(step.implementation),
        "state": normalize_for_hash(step.state),
        "step_args": normalize_for_hash(step.step_args),
    }
    return stable_hash(step_dict, digest_size=digest_size)


def get_hash_preprocessing_plan(plan: Any, digest_size: int = 16) -> str:
    """
    Hash a preprocessing plan as an ordered sequence of step hashes.

    Args:
        plan: Object with a `steps` sequence.
        digest_size: Blake2b digest size in bytes.

    Returns:
        Hexadecimal hash string.
    """
    steps = [preprocessing_step_hash(step) for step in plan.steps]
    return stable_hash(steps, digest_size=digest_size)


def ndarray_state_fingerprint(array: Any, digest_size: int = 16) -> dict[str, Any]:
    """
    Represent a NumPy/CuPy array by metadata and a full-content hash.

    Args:
        array: NumPy or CuPy array.
        digest_size: Blake2b digest size in bytes.

    Returns:
        JSON-compatible array fingerprint.
    """
    array = to_numpy(array)
    return {
        "kind": "ndarray",
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "hash": _blake_hash_bytes(_array_to_bytes(array), digest_size=digest_size),
    }


def tensor_state_fingerprint(tensor: torch.Tensor, digest_size: int = 16) -> dict[str, Any]:
    """
    Represent a torch tensor by metadata and a full-content hash.

    Args:
        tensor: Tensor to fingerprint.
        digest_size: Blake2b digest size in bytes.

    Returns:
        JSON-compatible tensor fingerprint.
    """
    return {
        "kind": "tensor",
        "shape": tuple(int(dim) for dim in tensor.shape),
        "dtype": str(tensor.dtype),
        "hash": _blake_hash_bytes(_tensor_to_bytes(tensor), digest_size=digest_size),
    }
