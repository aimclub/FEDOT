from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data import TensorData


def _canonical_json(obj: Any) -> str:
    """Serialize an object to canonical JSON after hash-specific normalization."""
    return json.dumps(
        normalize_for_hash(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _stable_bytes(obj: Any) -> bytes:
    """Return canonical UTF-8 bytes for a Python object."""
    return _canonical_json(obj).encode("utf-8")


def stable_hash(obj: Any, *, digest_size: int = 16) -> str:
    """
    Return a stable hexadecimal hash for normalized Python objects.

    Common runtime values such as tensors, arrays, enums, dataclasses and paths
    are converted to JSON-compatible values before hashing.
    """
    data = _stable_bytes(obj)
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def _blake_hash_bytes(data: bytes, digest_size: int = 16) -> str:
    """Return a Blake2b hash for already prepared bytes."""
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def _hash_to_int(data: bytes) -> int:
    """Map bytes to an unsigned integer using a small Blake2b digest."""
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def deterministic_positions(
    total_rows: int,
    n_samples: int,
    seed_data: dict[str, Any],
) -> list[int]:
    """
    Select deterministic row positions from the first axis.

    The same `total_rows`, `n_samples` and `seed_data` combination always
    produces the same sorted positions.
    """
    if total_rows <= 0 or n_samples <= 0:
        return []

    n_samples = min(n_samples, total_rows)

    if n_samples == 1:
        return [0]

    positions = {0, total_rows // 2, total_rows - 1}

    grid_count = max(1, n_samples // 2)
    if grid_count == 1:
        positions.add(total_rows // 2)
    else:
        for i in range(grid_count):
            pos = round(i * (total_rows - 1) / (grid_count - 1))
            positions.add(pos)

    seed_bytes = _stable_bytes(seed_data)
    counter = 0
    while len(positions) < n_samples:
        candidate = _hash_to_int(seed_bytes + counter.to_bytes(8, "little")) % total_rows
        positions.add(candidate)
        counter += 1

    return sorted(positions)


def sample_row_positions(
    shape: tuple[int, ...],
    dtype: Any,
    min_rows: int = 8,
    max_rows: int = 64,
) -> list[int]:
    """
    Build deterministic row positions for row-oriented feature containers.

    Args:
        shape: Shape of the feature matrix/tensor.
        dtype: Feature dtype included in the sampling seed.
        min_rows: Lower bound for the requested sample size.
        max_rows: Upper bound for the requested sample size.

    Returns:
        Sorted row indices selected from the first axis.
    """
    if not shape:
        return []

    n_rows = int(shape[0])
    n_sample_rows = min(max_rows, max(min_rows, int(np.sqrt(max(n_rows, 1)))))
    seed_data = {
        "shape": tuple(int(dim) for dim in shape),
        "dtype": str(dtype),
    }

    return deterministic_positions(
        total_rows=n_rows,
        n_samples=n_sample_rows,
        seed_data=seed_data,
    )


def _to_numpy(array: Any) -> np.ndarray:
    """Convert arrays from the active backend to a NumPy array for byte hashing."""
    backend = Backend()

    if backend.device.type != "cpu":
        return backend.xp.asnumpy(array)
    return np.asarray(array)


def _array_to_bytes(array: Any) -> bytes:
    """
    Convert an array fragment to stable bytes.

    Numeric arrays are hashed by raw contiguous bytes. Object arrays are routed
    through canonical JSON to support strings/categories/mixed values.
    """
    array = _to_numpy(array)

    if array.dtype.hasobject:
        return _stable_bytes(array.tolist())

    if np.issubdtype(array.dtype, np.floating):
        array = np.array(array, copy=True)
        array[np.isnan(array)] = np.nan

    return np.ascontiguousarray(array).tobytes()


def get_hash_raw_features(features: Any, row_positions: list[int], digest_size: int = 16) -> str:
    """
    Hash raw NumPy/CuPy feature metadata and sampled rows.

    Args:
        features: Raw feature array.
        row_positions: Row positions selected from the first axis.
        digest_size: Blake2b digest size in bytes.

    Returns:
        Hexadecimal hash string.
    """
    metadata = {
        "shape": tuple(int(dim) for dim in features.shape),
        "dtype": str(features.dtype),
    }

    h = hashlib.blake2b(digest_size=digest_size)
    h.update(_stable_bytes(metadata))

    if row_positions:
        h.update(_array_to_bytes(features[row_positions]))

    return h.hexdigest()


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a tensor or tensor fragment to stable bytes.

    Sparse tensors are densified, and only the supplied tensor is moved to CPU.
    """
    tensor = tensor.detach()

    if tensor.is_sparse:
        tensor = tensor.to_dense()

    tensor = tensor.cpu().contiguous()

    if torch.is_floating_point(tensor):
        tensor = tensor.clone()
        tensor[torch.isnan(tensor)] = float("nan")

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
        "ts_orientation": td.ts_orientation,
        "ts_terms_idx": td.ts_terms_idx,
        "ts_forecast_horizon": td.ts_forecast_horizon,
        "ts_init_shape": td.ts_init_shape,
    }


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

    return h.hexdigest()


def implementation_signature(implementation: Any) -> Optional[dict[str, Any]]:
    """Return a stable class identifier for a preprocessing implementation."""
    if implementation is None:
        return None

    cls = implementation if inspect.isclass(implementation) else implementation.__class__

    return {
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
    }


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


DEFAULT_EXCLUDED_MODEL_FIELDS = {
    "logger",
    "cache",
    "cacher",
    "device_context",
}


def normalize_for_hash(obj: Any) -> Any:
    """
    Convert arbitrary runtime state to a stable JSON-compatible structure.

    Large tensors and arrays are represented by metadata plus a content hash
    instead of expanding them to nested lists.
    """
    if obj is None:
        return None

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, torch.device):
        return obj.type

    if isinstance(obj, torch.dtype):
        return str(obj)

    if isinstance(obj, np.dtype):
        return str(obj)

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        return tensor_state_fingerprint(obj)

    if isinstance(obj, np.ndarray) or type(obj).__module__.split(".", maxsplit=1)[0] == "cupy":
        return ndarray_state_fingerprint(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, tuple):
        return [normalize_for_hash(x) for x in obj]

    if isinstance(obj, list):
        return [normalize_for_hash(x) for x in obj]

    if isinstance(obj, set):
        return sorted(normalize_for_hash(x) for x in obj)

    if isinstance(obj, dict):
        normalized_items = {}
        for key, value in obj.items():
            normalized_key = str(normalize_for_hash(key))
            normalized_items[normalized_key] = normalize_for_hash(value)
        return {key: normalized_items[key] for key in sorted(normalized_items)}

    if is_dataclass(obj):
        return normalize_for_hash(asdict(obj))

    cls = obj.__class__
    return {
        "__object__": f"{cls.__module__}.{cls.__qualname__}",
        "__str__": str(obj),
    }


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


def ndarray_state_fingerprint(array: Any, digest_size: int = 16) -> dict[str, Any]:
    """
    Represent a NumPy/CuPy array by metadata and a full-content hash.

    Args:
        array: NumPy or CuPy array.
        digest_size: Blake2b digest size in bytes.

    Returns:
        JSON-compatible array fingerprint.
    """
    array = _to_numpy(array)
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
