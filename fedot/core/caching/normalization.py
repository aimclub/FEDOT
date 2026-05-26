from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional, Set

import numpy as np
import torch

from fedot.core.data.common.types import ARRAY_RUNTIME_TYPES


def normalize_for_hash(obj: Any, _seen: Optional[Set[int]] = None) -> Any:
    """
    Convert arbitrary runtime state to a stable JSON-compatible structure.

    Large tensors and arrays are represented by metadata plus a content hash
    instead of expanding them to nested lists.
    """
    if _seen is None:
        _seen = set()

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
        from fedot.core.caching.fingerprints import tensor_state_fingerprint

        return tensor_state_fingerprint(obj)

    if isinstance(obj, ARRAY_RUNTIME_TYPES):
        from fedot.core.caching.fingerprints import ndarray_state_fingerprint

        return ndarray_state_fingerprint(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, tuple):
        return _normalize_sequence_for_hash(obj, _seen)

    if isinstance(obj, list):
        return _normalize_sequence_for_hash(obj, _seen)

    if isinstance(obj, set):
        return sorted(normalize_for_hash(x, _seen) for x in obj)

    if isinstance(obj, dict):
        _check_cycle(obj, _seen)
        normalized_items = {}
        try:
            for key, value in obj.items():
                normalized_key = str(normalize_for_hash(key, _seen))
                normalized_items[normalized_key] = normalize_for_hash(value, _seen)
            return {key: normalized_items[key] for key in sorted(normalized_items)}
        finally:
            _seen.remove(id(obj))

    if is_dataclass(obj) and not inspect.isclass(obj):
        _check_cycle(obj, _seen)
        try:
            return {
                field.name: normalize_for_hash(getattr(obj, field.name), _seen)
                for field in fields(obj)
            }
        finally:
            _seen.remove(id(obj))

    cls = obj.__class__
    raise TypeError(
        f"Unsupported type for cache hashing: {cls.__module__}.{cls.__qualname__}"
    )


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


def _check_cycle(obj: Any, seen: Set[int]) -> None:
    obj_id = id(obj)
    if obj_id in seen:
        raise TypeError(f"Cycle detected during cache hash normalization: {type(obj)}")
    seen.add(obj_id)


def _normalize_sequence_for_hash(obj: Any, seen: Set[int]) -> list[Any]:
    _check_cycle(obj, seen)
    try:
        return [normalize_for_hash(x, seen) for x in obj]
    finally:
        seen.remove(id(obj))
