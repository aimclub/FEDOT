import inspect
from typing import Any

from fedot.core.common.registry import Registry
from fedot.core.caching.fingerprints import (
    get_hash_preprocessing_plan,
    get_hash_raw_features,
    get_hash_tensordata,
    get_model_attributes,
)
from fedot.core.caching.normalization import stable_hash
from fedot.core.caching.sampling import (
    sample_row_positions,
)
from fedot.core.caching.rules import HasherNotFoundError
from fedot.core.data.common.types import ARRAY_RUNTIME_TYPES
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
from fedot.preprocessing.planner import PreprocessingPlan
from fedot.core.common.registry_predicates import *


class Hasher(Registry):
    """Registry-based dispatcher for cache fingerprint builders."""

    not_found_error = HasherNotFoundError
    not_found_message = 'No hashing function registered for data type: {source_type}'

    @classmethod
    def hash(cls, data: Any, **kwargs: Any) -> str:
        """
        Hash a supported object using the first matching registered creator.

        Args:
            data: Object to fingerprint.
            **kwargs: Optional arguments forwarded to a concrete hash function.

        Returns:
            Hexadecimal hash string.

        Raises:
            HasherNotFoundError: If no hash function is registered for `data`.
        """
        hashing_func = cls.resolve_creator(data)
        return hashing_func(data, **kwargs)


@Hasher.register_creator(is_array_runtime)
def raw_fingerprint(
    data: Any,
    target: Any = None,
    min_rows: int = 8,
    max_rows: int = 64,
    digest_size: int = 16,
) -> str:
    """
    Build a fast fingerprint for a raw NumPy/CuPy feature array.

    The hash combines array shape, dtype and a deterministic sample of rows from
    the first axis. It is intended for row-oriented feature containers.
    """
    row_positions = sample_row_positions(
        shape=tuple(data.shape),
        dtype=data.dtype,
        min_rows=min_rows,
        max_rows=max_rows,
    )
    return get_hash_raw_features(
        features=data,
        row_positions=row_positions,
        target=target,
        digest_size=digest_size,
    )


@Hasher.register_creator(is_tensor_data)
def ready_fingerprint(
    td: TensorData,
    min_rows: int = 8,
    max_rows: int = 64,
    digest_size: int = 16,
) -> str:
    """
    Build a fingerprint for prepared `TensorData`.

    The hash combines selected `TensorData` metadata with a deterministic sample
    of `td.features` rows. Target and prediction values are not sampled.
    """
    if td.features is None:
        raise ValueError("TensorData features are required for ready fingerprint.")

    return get_hash_tensordata(td, min_rows=min_rows, max_rows=max_rows, digest_size=digest_size)


@Hasher.register_creator(is_preprocessing_plan)
def preprocessing_plan_hash(plan: PreprocessingPlan, digest_size: int = 16) -> str:
    """
    Build a fingerprint for a preprocessing plan.

    Plan steps are hashed in order, including step configuration and fitted
    implementation state when it is present.
    """
    return get_hash_preprocessing_plan(plan, digest_size=digest_size)


@Hasher.register_creator(is_preprocessing_handler)
def preprocessing_model_hash(model: Any, digest_size: int = 16) -> str:
    """
    Build a fingerprint for a preprocessing handler class or fitted instance.

    For fitted instances, the hash includes the class path and normalized
    attributes from `__dict__`. For classes, only the class path is used.
    """
    attributes = get_model_attributes(model)
    return stable_hash(attributes, digest_size=digest_size)


def _validate_loaded_hash(data: Any, expected_hash: str = None) -> None:
    if expected_hash is None:
        return

    from fedot.core.caching.hasher import Hasher

    actual_hash = Hasher.hash(data)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Loaded cache hash mismatch: expected {expected_hash}, got {actual_hash}"
        )