import hashlib
from typing import Any

import numpy as np

from fedot.core.caching.normalization import _stable_bytes


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
