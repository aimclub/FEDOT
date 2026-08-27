import math
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Union

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import drop_rows_with_nan, flatten_if_needed
from fedot.core.operations.evaluation.operation_implementations.rules import (
    SpectrumNComponentsMethod,
    is_auto_n_components,
    is_integral_number,
    is_real_number,
    is_spectrum_n_components_method,
    pca_mle_shape_error_message,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_broken_stick_n,
    validate_decomposition_fit_samples,
    validate_spectrum_rank_selection,
)
from fedot.validation.errors import FedotValidationError


def prepare_finite_features(
    features: torch.Tensor,
    log: Any,
    op_name: str,
    *,
    require_min_samples: bool = True,
) -> torch.Tensor:
    """Flatten features and drop rows that contain NaN.

    Args:
        features: Feature tensor.
        log: Logger used for the NaN-drop warning.
        op_name: Operation name in validation / warning messages.
        require_min_samples: If True, require at least two finite rows.

    Returns:
        Finite feature rows only.
    """
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
    """Replace features in ``data`` with projected values.

    Args:
        data: Source TensorData.
        projected: Projected feature matrix.

    Returns:
        Updated TensorData (numerical indices reset).
    """
    return replace(
        data,
        features=projected,
        categorical_idx=[],
        numerical_idx=list(range(projected.shape[1])),
        features_names=None,
        fingerprint=None,
    )


def max_decomposition_rank(n_samples: int, n_features: int) -> int:
    """Return SVD rank bound ``max(min(n_samples, n_features), 1)``.

    Args:
        n_samples: Number of finite training rows.
        n_features: Feature width.

    Returns:
        Maximum feasible number of components.
    """
    return max(min(n_samples, n_features), 1)


def default_components_budget(n_samples: int, n_features: int) -> int:
    """Default ``k`` for repository ``n_components='auto'``.

    ``max(1, min(rank, n_features // 2))``. It is a width cap so wide
    tables (e.g. after OHE) compress even when a variance quota would keep
    almost every column.

    Args:
        n_samples: Number of finite training rows.
        n_features: Feature width.

    Returns:
        ``max(1, min(rank, n_features // 2))``.
    """
    rank = max_decomposition_rank(n_samples, n_features)
    half_features = max(n_features // 2, 1)
    return max(1, min(rank, half_features))


def broken_stick_expectations(n: int, *, device=None, dtype=None) -> torch.Tensor:
    """Broken-stick expected shares for ``n`` ordered components.

    Args:
        n: Number of spectrum pieces (``n >= 1``).
        device: Torch device for the result.
        dtype: Torch dtype for the result.

    Returns:
        Tensor of length ``n`` with expected shares (sums to 1).
    """
    n = validate_broken_stick_n(n)
    # b_k = (1/n) * sum_{i=k}^{n} (1/i), k = 1..n
    inv = 1.0 / torch.arange(1, n + 1, device=device, dtype=dtype or torch.float32)
    suffix = torch.flip(torch.cumsum(torch.flip(inv, dims=(0,)), dim=0), dims=(0,))
    return suffix / n


def n_components_from_broken_stick(proportions: torch.Tensor) -> int:
    """Select leading components above broken-stick expectations.

    Args:
        proportions: Spectrum shares (renormalized to sum 1 if needed).

    Returns:
        Number of leading components to keep (at least 1).
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
    # Keep the first contiguous run from the start.
    stop = (~exceed).nonzero(as_tuple=False)
    if stop.numel() == 0:
        return n
    return max(1, int(stop[0].item()))


def n_components_from_elbow(spectrum: torch.Tensor) -> int:
    """Select rank by elbow (max distance to the first–last chord).

    Args:
        spectrum: Decreasing singular values (or similar scree curve).

    Returns:
        ``knee_index + 1`` (includes the elbow point), at least 1.
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


def _assess_mle_dimension(spectrum: torch.Tensor, rank: int, n_samples: int) -> float:
    """Log-likelihood of PCA rank ``rank`` (Minka, NIPS 2000).

    Port of ``sklearn.decomposition._pca._assess_dimension``. ``spectrum`` is
    the explained-variance vector (eigenvalues), sorted descending.
    """
    n_features = int(spectrum.shape[0])
    eps = 1e-15
    if float(spectrum[rank - 1]) < eps:
        return float('-inf')

    pu = -rank * math.log(2.0)
    for i in range(1, rank + 1):
        pu += (
            math.lgamma((n_features - i + 1) / 2.0)
            - math.log(math.pi) * (n_features - i + 1) / 2.0
        )

    pl = float(torch.sum(torch.log(spectrum[:rank])))
    pl = -pl * n_samples / 2.0

    v = max(eps, float(torch.sum(spectrum[rank:])) / (n_features - rank))
    pv = -math.log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = math.log(2.0 * math.pi) * (m + rank) / 2.0

    pa = 0.0
    spectrum_ = spectrum.clone()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, n_features):
            pa += math.log(
                float(
                    (spectrum[i] - spectrum[j])
                    * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
                )
            ) + math.log(n_samples)

    return pu + pl + pv + pp - pa / 2.0 - rank * math.log(n_samples) / 2.0


def n_components_from_mle(spectrum: torch.Tensor, n_samples: int) -> int:
    """Select PCA rank by Minka's MLE (sklearn ``n_components='mle'``).

    Args:
        spectrum: Explained-variance eigenvalues, shape ``(n_features,)``.
        n_samples: Number of training rows (must be ``>= n_features``).

    Returns:
        Rank in ``[1, n_features - 1]``.
    """
    values = spectrum.detach().flatten().to(dtype=torch.float64)
    n_spectrum = int(values.numel())
    if n_spectrum <= 1:
        return 1

    ll = torch.full((n_spectrum,), float('-inf'), dtype=torch.float64)
    for rank in range(1, n_spectrum):
        ll[rank] = _assess_mle_dimension(values, rank, n_samples)
    chosen = int(torch.argmax(ll).item())
    return max(1, min(chosen, n_spectrum - 1))


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
    """Resolve ``elbow`` / ``broken_stick`` to a feasible component count.

    Args:
        method: Spectrum rank-selection method.
        singular_values: Singular values (preferred for elbow; used as ``S^2``
            for broken stick when proportions are absent).
        proportions: Explained-variance shares for broken stick.
        max_components: Upper bound from data rank.

    Returns:
        Selected number of components in ``[1, max_components]``.
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
        # Prefer ready proportions; otherwise use eigenvalue proxies S^2.
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
    """Resolve PCA ``n_components`` to an integer rank.

    Args:
        n_components: Int, variance ratio in ``(0, 1]`` (``1.0`` = full rank),
            ``auto``, Minka ``mle``, ``elbow``, or ``broken_stick``.
        n_samples: Number of finite training rows.
        n_features: Feature width.
        explained_variance_ratio: Full explained-variance shares.
        singular_values: Singular values for spectrum methods.

    Returns:
        Feasible number of components.
    """
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

    if isinstance(n_components, str) and n_components == 'mle':
        if n_samples < n_features:
            raise FedotValidationError(pca_mle_shape_error_message())
        if singular_values is None:
            raise FedotValidationError("n_components='mle' requires singular_values")
        spectrum = singular_values.square() / max(n_samples - 1, 1)
        k = n_components_from_mle(spectrum, n_samples)
        return max(1, min(k, max_components))

    # Non-integral float in (0, 1]: explained-variance ratio.
    # 1.0 maps to full rank (all variance), unlike sklearn where 1.0 == 1.
    if is_real_number(n_components) and not is_integral_number(n_components):
        ratio = float(n_components)
        if ratio <= 1.0:
            if ratio == 1.0:
                return max_components
            cumsum = torch.cumsum(explained_variance_ratio, dim=0)
            hits = (cumsum >= ratio).nonzero(as_tuple=False)
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
    """Resolve TruncatedSVD ``n_components`` to an integer rank.

    Args:
        n_components: Int, feature-fraction in ``(0, 1]``, ``auto``, ``elbow``,
            or ``broken_stick``.
        n_samples: Number of finite training rows.
        n_features: Feature width.
        singular_values: Required for spectrum methods.

    Returns:
        Feasible number of components.
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

    # Non-integral float in (0, 1]: feature fraction (not variance ratio).
    if is_real_number(n_components) and not is_integral_number(n_components):
        fraction = float(n_components)
        if fraction <= 1.0:
            resolved = int(round(fraction * n_features))
            return max(1, min(resolved, max_components))

    return max(1, min(int(n_components), max_components))


def require_fitted_feature_width(
    features: torch.Tensor,
    n_features: Optional[int],
    op_name: str,
) -> None:
    """Raise if transform width does not match fit.

    Args:
        features: Flattened feature matrix.
        n_features: Width stored at fit, or ``None`` if unknown.
        op_name: Operation name in the error message.

    Raises:
        FedotValidationError: When ``features.shape[1] != n_features``.
    """
    if n_features is None:
        return
    width = int(features.shape[1])
    if width != n_features:
        raise FedotValidationError(
            f'{op_name} expected {n_features} features, got {width}'
        )


def project_with_components(
    features: torch.Tensor,
    components: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project features onto component rows, optionally after centering.

    Args:
        features: Feature matrix.
        components: Component matrix with shape ``(k, n_features)``.
        mean: Optional feature mean (PCA). If None, no centering.

    Returns:
        Projected features of shape ``(n_samples, k)``.
    """
    components = components.to(device=features.device, dtype=features.dtype)
    if mean is not None:
        features = features - mean.to(device=features.device, dtype=features.dtype)
    return features @ components.T
