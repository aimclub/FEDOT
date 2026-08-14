from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Union

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import drop_rows_with_nan, flatten_if_needed
from fedot.core.operations.evaluation.operation_implementations.rules import (
    SpectrumNComponentsMethod,
    is_auto_n_components,
    is_spectrum_n_components_method,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_broken_stick_n,
    validate_decomposition_fit_samples,
    validate_spectrum_rank_selection,
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


def broken_stick_expectations(n: int, *, device=None, dtype=None) -> torch.Tensor:
    """Broken-stick share expectations for ``n`` ordered components.

    ``b_k = (1 / n) * sum_{i=k}^{n} (1 / i)`` for ``k = 1..n`` (1-based).
    """
    n = validate_broken_stick_n(n)
    inv = 1.0 / torch.arange(1, n + 1, device=device, dtype=dtype or torch.float32)
    # suffix[k] = sum_{i=k}^{n-1} inv[i] = sum_{j=k+1}^{n} 1/j
    suffix = torch.flip(torch.cumsum(torch.flip(inv, dims=(0,)), dim=0), dims=(0,))
    return suffix / n


def n_components_from_broken_stick(proportions: torch.Tensor) -> int:
    """Keep leading components whose share exceeds the broken-stick model.

    ``proportions`` should be non-increasing explained-variance shares (or any
    positive spectrum that will be renormalized to sum to 1).
    """
    values = proportions.detach().flatten()
    n = int(values.numel())
    if n <= 1:
        return max(n, 1)

    total = float(values.sum().item())
    if total <= 0:
        return 1
    shares = values / values.sum()
    expected = broken_stick_expectations(n, device=shares.device, dtype=shares.dtype)
    exceed = shares > expected
    # First contiguous run from the start; always keep at least one component.
    stop = (~exceed).nonzero(as_tuple=False)
    if stop.numel() == 0:
        return n
    return max(1, int(stop[0].item()))


def n_components_from_elbow(spectrum: torch.Tensor) -> int:
    """Elbow (knee) cut: max distance from the chord joining first and last SV.

    Returns ``knee_index + 1`` (include the elbow point), at least 1.
    """
    values = spectrum.detach().flatten()
    n = int(values.numel())
    if n <= 1:
        return max(n, 1)

    x = torch.arange(n, device=values.device, dtype=values.dtype)
    coords = torch.stack((x, values), dim=1)
    line = coords[-1] - coords[0]
    line_norm = torch.linalg.norm(line)
    if float(line_norm.item()) == 0.0:
        return 1

    line_unit = line / line_norm
    from_first = coords - coords[0]
    parallel = torch.outer(from_first @ line_unit, line_unit)
    dist = torch.linalg.norm(from_first - parallel, dim=1)
    knee_idx = int(torch.argmax(dist).item())
    return max(1, knee_idx + 1)


SPECTRUM_N_COMPONENTS: Dict[SpectrumNComponentsMethod, Callable[[torch.Tensor], int]] = {
    SpectrumNComponentsMethod.ELBOW: n_components_from_elbow,
    SpectrumNComponentsMethod.BROKEN_STICK: n_components_from_broken_stick,
}


def resolve_spectrum_n_components(
    method: Union[str, SpectrumNComponentsMethod],
    *,
    singular_values: Optional[torch.Tensor] = None,
    proportions: Optional[torch.Tensor] = None,
    max_components: int,
) -> int:
    """Map ``elbow`` / ``broken_stick`` + spectrum → feasible ``k``.

    Prefer ``singular_values`` for elbow. For ``broken_stick``,
    ``proportions`` (already normalized shares) may be used instead of ``S^2``.
    """
    validated = validate_spectrum_rank_selection(
        method,
        singular_values=singular_values,
        proportions=proportions,
        max_components=max_components,
    )
    method_enum: SpectrumNComponentsMethod = validated['method']
    sv = validated['singular_values']
    props = validated['proportions']

    if method_enum is SpectrumNComponentsMethod.ELBOW:
        spectrum = sv if sv is not None else props
    else:
        spectrum = props if props is not None else sv.square()

    k = SPECTRUM_N_COMPONENTS[method_enum](spectrum)
    return max(1, min(k, validated['max_components'], int(spectrum.numel())))


def resolve_pca_n_components(
    n_components: Union[int, float, str, SpectrumNComponentsMethod, None],
    *,
    n_samples: int,
    n_features: int,
    explained_variance_ratio: torch.Tensor,
    singular_values: Optional[torch.Tensor] = None,
) -> int:
    """Resolve PCA ``n_components`` (auto / int / variance / mle / spectrum) to ``k``."""
    max_components = max_decomposition_rank(n_samples, n_features)

    if is_auto_n_components(n_components):
        return default_components_budget(n_samples, n_features)

    if is_spectrum_n_components_method(n_components):
        return resolve_spectrum_n_components(
            n_components,
            singular_values=singular_values,
            proportions=explained_variance_ratio,
            max_components=max_components,
        )

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


def resolve_truncated_svd_n_components(
    n_components: Union[int, float, str, SpectrumNComponentsMethod, None],
    *,
    n_samples: int,
    n_features: int,
    singular_values: Optional[torch.Tensor] = None,
) -> int:
    """Resolve TruncatedSVD ``n_components`` to feasible ``k``.

    - ``auto`` / unset → half-feature budget
    - float in ``(0, 1]`` → fraction of ``n_features`` (not explained variance)
    - ``elbow`` / ``broken_stick`` → spectrum methods (require ``singular_values``)
    - positive int → clamped to SVD rank
    """
    max_components = max_decomposition_rank(n_samples, n_features)

    if is_auto_n_components(n_components):
        return default_components_budget(n_samples, n_features)

    if is_spectrum_n_components_method(n_components):
        return resolve_spectrum_n_components(
            n_components,
            singular_values=singular_values,
            max_components=max_components,
        )

    if isinstance(n_components, float) and n_components <= 1.0:
        resolved = int(round(n_components * n_features))
        return max(1, min(resolved, max_components))

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
