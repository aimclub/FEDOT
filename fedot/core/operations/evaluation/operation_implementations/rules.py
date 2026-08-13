from typing import Any

from fedot.core.operations.operation_parameters import get_default_params

PCA_OPERATION_TYPE = 'pca'
PCA_SUPPORTED_N_COMPONENTS_STR = frozenset({'mle'})
PCA_MIN_FIT_SAMPLES = 2
PCA_MIN_THRESHOLD_TS = 2


def default_pca_n_components() -> Any:
    """Default ``n_components`` from ``default_operation_params.json``."""
    return get_default_params(PCA_OPERATION_TYPE).get('n_components')


def is_valid_pca_n_components(value: Any) -> bool:
    """Whether ``n_components`` is an allowed PCA hyperparameter value."""
    if isinstance(value, str):
        return value in PCA_SUPPORTED_N_COMPONENTS_STR
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return value >= 1
    if isinstance(value, float):
        if value < 1.0:
            return value > 0.0
        return True
    return False


def pca_n_components_error_message(value: Any) -> str:
    return (
        f"Unsupported PCA n_components: {value!r}. "
        f"Expected positive int, float in (0, 1], or one of "
        f"{sorted(PCA_SUPPORTED_N_COMPONENTS_STR)}."
    )


def has_enough_pca_fit_samples(n_samples: Any) -> bool:
    return isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= PCA_MIN_FIT_SAMPLES


def pca_fit_samples_error_message(n_samples: Any) -> str:
    return (
        f'PCA fit needs at least {PCA_MIN_FIT_SAMPLES} finite samples, '
        f'got {n_samples} after dropping rows with NaN'
    )
