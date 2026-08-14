from dataclasses import replace
from typing import Any, Optional, Union

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import drop_rows_with_nan, flatten_if_needed
from fedot.core.operations.evaluation.operation_implementations.rules import is_auto_n_components
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_decomposition_fit_samples,
)


def prepare_finite_features(
    features: torch.Tensor,
    log: Any,
    op_name: str,
    *,
    require_min_samples: bool = True,
) -> torch.Tensor:
    """Flatten features, drop NaN rows, optionally enforce min sample count."""
    features = flatten_if_needed(features)
    clean, n_dropped = drop_rows_with_nan(features)
    if n_dropped:
        log.warning(
            f'{op_name} fit: dropping {n_dropped} sample(s) with NaN; '
            f'they are not used for fitting. Transform still returns all rows '
            f'(NaN inputs remain NaN after projection).'
        )
    if require_min_samples:
        validate_decomposition_fit_samples(clean.shape[0], op_name=op_name)
    return clean


def replace_projected_features(data: TensorData, projected: torch.Tensor) -> TensorData:
    """Write projected features back into a TensorData container."""
    return replace(
        data,
        features=projected,
        categorical_idx=[],
        numerical_idx=list(range(projected.shape[1])),
        features_names=None,
        fingerprint=None,
    )


def max_decomposition_rank(n_samples: int, n_features: int) -> int:
    """Hard SVD rank bound ``min(n_samples, n_features)`` (at least 1)."""
    return max(min(n_samples, n_features), 1)


def default_components_budget(n_samples: int, n_features: int) -> int:
    """Data-dependent default ``k`` when ``n_components='auto'`` / unset.

    Uses at most half of the feature width, still capped by SVD rank:

    ``k = max(1, min(rank, n_features // 2))``

    This is a practical default for both PCA and TruncatedSVD: keeps compression
    meaningful on wide tables (e.g. after OHE) without requiring a variance
    target, and avoids keeping nearly full rank.
    """
    rank = max_decomposition_rank(n_samples, n_features)
    half_features = max(n_features // 2, 1)
    return max(1, min(rank, half_features))


def resolve_pca_n_components(
    n_components: Union[int, float, str, None],
    *,
    n_samples: int,
    n_features: int,
    explained_variance_ratio: torch.Tensor,
) -> int:
    """Resolve PCA ``n_components`` (auto / int / variance ratio / ``mle``) to ``k``."""
    max_components = max_decomposition_rank(n_samples, n_features)

    if is_auto_n_components(n_components):
        return default_components_budget(n_samples, n_features)

    if isinstance(n_components, str):
        # Schema allows only ``mle`` among remaining strings.
        if n_samples < n_features:
            n_components = 0.5
        else:
            return max(1, max_components - 1) if max_components > 1 else 1

    if isinstance(n_components, float) and n_components < 1.0:
        cumsum = torch.cumsum(explained_variance_ratio, dim=0)
        hits = (cumsum >= n_components).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return max_components
        return int(hits[0].item()) + 1

    resolved = int(n_components)
    if resolved > max_components:
        resolved = max_components
    return max(1, resolved)


def resolve_int_n_components(
    n_components: Union[int, str, None],
    *,
    n_samples: int,
    n_features: int,
) -> int:
    """Resolve TruncatedSVD ``n_components`` (auto / positive int) to feasible ``k``."""
    if is_auto_n_components(n_components):
        return default_components_budget(n_samples, n_features)

    max_components = max_decomposition_rank(n_samples, n_features)
    return max(1, min(int(n_components), max_components))


def project_with_components(
    features: torch.Tensor,
    components: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project features with optional centering (PCA) onto ``components`` rows."""
    components = components.to(device=features.device, dtype=features.dtype)
    if mean is not None:
        features = features - mean.to(device=features.device, dtype=features.dtype)
    return features @ components.T
