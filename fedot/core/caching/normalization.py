from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional, Set, Union, get_args, get_origin

import numpy as np
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.common.types import ARRAY_RUNTIME_TYPES
from fedot.core.data.tensor_data import TensorData


TENSOR_DATA_CACHE_FORMAT = "fedot-tensor-data-cache"
TENSOR_DATA_CACHE_FORMATS = frozenset({
    TENSOR_DATA_CACHE_FORMAT,
    "fedot-tensor-data-cache-v1",
})


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

    if isinstance(obj, slice):
        return {
            "start": normalize_for_hash(obj.start, _seen),
            "stop": normalize_for_hash(obj.stop, _seen),
            "step": normalize_for_hash(obj.step, _seen),
        }

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


def _is_cupy_array(data: Any) -> bool:
    return "cupy" in type(data).__module__


def prepare_value_for_torch_save(value: Any) -> Any:
    """
    Convert runtime values to cache-safe values.

    Main rule:
    - torch.Tensor -> detached CPU tensor
    - cupy.ndarray -> numpy.ndarray
    - dataclass -> dict with normalized fields
    - list/tuple/dict -> recursively processed
    - primitive metadata -> JSON/torch-save friendly values
    """
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (str, int, float, bool, bytes)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, np.dtype):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if _is_cupy_array(value):
        return value.get()

    if isinstance(value, dict):
        return {
            prepare_value_for_torch_save(k): prepare_value_for_torch_save(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [prepare_value_for_torch_save(v) for v in value]

    if isinstance(value, tuple):
        return tuple(prepare_value_for_torch_save(v) for v in value)

    if isinstance(value, (set, frozenset)):
        return [prepare_value_for_torch_save(v) for v in value]

    if is_dataclass(value) and not inspect.isclass(value):
        return {
            "class_path": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": {
                field.name: prepare_value_for_torch_save(getattr(value, field.name))
                for field in fields(value)
            },
        }

    raise TypeError(
        "Unsupported type for torch cache saving: "
        f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    )


def build_tensor_data_payload(td: Any) -> dict[str, Any]:
    """
    Build TensorData payload automatically from dataclass fields.

    This avoids hardcoding TensorData field names.
    """
    if not is_dataclass(td):
        raise TypeError(f"Expected dataclass TensorData, got {type(td)}")

    payload_fields = {}

    for field in fields(td):
        value = getattr(td, field.name)
        payload_fields[field.name] = prepare_value_for_torch_save(value)

    return {
        "format": TENSOR_DATA_CACHE_FORMAT,
        "class_path": f"{td.__class__.__module__}.{td.__class__.__qualname__}",
        "fields": payload_fields,
    }


def _prepare_preprocessing_state_value(value: Any) -> Any:
    """Best-effort state preparation that keeps arbitrary custom attributes pickleable."""
    if isinstance(value, dict):
        return {
            _prepare_preprocessing_state_value(key): _prepare_preprocessing_state_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_prepare_preprocessing_state_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_prepare_preprocessing_state_value(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return [_prepare_preprocessing_state_value(item) for item in value]

    try:
        return prepare_value_for_torch_save(value)
    except TypeError:
        return value


def build_preprocessing_model_payload(data: Any) -> Any:
    """Prepare fitted preprocessing state for pickle without mutating the source model."""
    if inspect.isclass(data) or not hasattr(data, "__dict__"):
        return data

    payload = data.__class__.__new__(data.__class__)
    prepared_state = {}

    for field_name, field_value in data.__dict__.items():
        prepared_state[field_name] = _prepare_preprocessing_state_value(field_value)

    payload.__dict__.update(prepared_state)
    return payload


def restore_tensor_data_payload(payload: dict[str, Any]) -> TensorData:
    """Restore `TensorData` from a normalized torch-save payload."""
    if isinstance(payload, TensorData):
        return payload.to(Backend().device)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Unsupported TensorData cache payload type: {type(payload).__name__}"
        )

    payload_format = payload.get("format")
    if payload_format not in TENSOR_DATA_CACHE_FORMATS:
        raise ValueError(
            f"Unsupported TensorData cache payload format: {payload_format!r}"
        )

    payload_fields = payload.get("fields")
    if not isinstance(payload_fields, dict):
        raise ValueError("TensorData cache payload must contain a fields mapping.")

    restored_fields = {}
    for field in fields(TensorData):
        if field.name in payload_fields:
            restored_fields[field.name] = _restore_value_for_field(
                payload_fields[field.name],
                field.type,
            )

    tensor_data = TensorData(**restored_fields)
    return tensor_data.to(Backend().device)


def prepare_loaded_preprocessing_model(data: Any) -> Any:
    """Move loaded preprocessing model tensor state to the active backend device."""
    return _move_tensors_to_backend(data)


def _restore_value_for_field(value: Any, expected_type: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value

    enum_type = _extract_enum_type(expected_type)
    if enum_type is not None and value is not None:
        return enum_type(value)

    if isinstance(value, dict) and "class_path" in value and "fields" in value:
        cls = _import_class(value["class_path"])
        dataclass_fields = value["fields"]
        if not is_dataclass(cls):
            return dataclass_fields
        return cls(**{
            field.name: _restore_value_for_field(
                dataclass_fields.get(field.name),
                field.type,
            )
            for field in fields(cls)
            if field.name in dataclass_fields
        })

    if isinstance(value, dict):
        return {
            _restore_value_for_field(key): _restore_value_for_field(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_restore_value_for_field(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_restore_value_for_field(item) for item in value)

    return value


def _extract_enum_type(expected_type: Any) -> Optional[type[Enum]]:
    if _is_enum_type(expected_type):
        return expected_type

    origin = get_origin(expected_type)
    if origin is Union:
        for arg in get_args(expected_type):
            if _is_enum_type(arg):
                return arg

    return None


def _is_enum_type(value: Any) -> bool:
    return inspect.isclass(value) and issubclass(value, Enum)


def _import_class(class_path: str) -> type:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class path in cache payload: {class_path}")

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _move_tensors_to_backend(value: Any) -> Any:
    device = Backend().device

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            _move_tensors_to_backend(key): _move_tensors_to_backend(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_move_tensors_to_backend(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_move_tensors_to_backend(item) for item in value)

    if isinstance(value, set):
        return {_move_tensors_to_backend(item) for item in value}

    if isinstance(value, frozenset):
        return frozenset(_move_tensors_to_backend(item) for item in value)

    if hasattr(value, "__dict__") and not inspect.isclass(value):
        for field_name, field_value in value.__dict__.items():
            setattr(value, field_name, _move_tensors_to_backend(field_value))

    return value
