from typing import Any

PCA_OPERATION_TYPE = 'pca'
TRUNCATED_SVD_OPERATION_TYPE = 'truncated_svd'
PCA_SUPPORTED_N_COMPONENTS_STR = frozenset({'mle', 'auto'})
TRUNCATED_SVD_SUPPORTED_N_COMPONENTS_STR = frozenset({'auto'})
DECOMPOSITION_MIN_FIT_SAMPLES = 2
AUTO_N_COMPONENTS_ALIASES = frozenset({None, 'auto'})


def is_auto_n_components(value: Any) -> bool:
    """Whether ``n_components`` means data-dependent default (half-rank budget)."""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'auto':
        return True
    return False


def is_valid_pca_n_components(value: Any) -> bool:
    """Whether ``n_components`` is an allowed PCA hyperparameter value."""
    if is_auto_n_components(value):
        return True
    if isinstance(value, str):
        return value in PCA_SUPPORTED_N_COMPONENTS_STR
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 1
    if isinstance(value, float):
        if value < 1.0:
            return value > 0.0
        return True
    return False


def is_valid_truncated_svd_n_components(value: Any) -> bool:
    """TruncatedSVD: positive int or ``auto`` (half-rank budget)."""
    if is_auto_n_components(value):
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def pca_n_components_error_message(value: Any) -> str:
    return (
        f"Unsupported PCA n_components: {value!r}. "
        f"Expected positive int, float in (0, 1], or one of "
        f"{sorted(PCA_SUPPORTED_N_COMPONENTS_STR)}."
    )


def truncated_svd_n_components_error_message(value: Any) -> str:
    return (
        f"Unsupported TruncatedSVD n_components: {value!r}. "
        f"Expected a positive int or 'auto'."
    )


def has_enough_decomposition_fit_samples(n_samples: Any) -> bool:
    return (
        isinstance(n_samples, int)
        and not isinstance(n_samples, bool)
        and n_samples >= DECOMPOSITION_MIN_FIT_SAMPLES
    )


def decomposition_fit_samples_error_message(n_samples: Any, op_name: str = 'decomposition') -> str:
    return (
        f'{op_name} fit needs at least {DECOMPOSITION_MIN_FIT_SAMPLES} finite samples, '
        f'got {n_samples} after dropping rows with NaN'
    )
