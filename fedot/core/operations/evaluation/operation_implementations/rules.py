import numbers
from enum import Enum
from typing import Any

PCA_OPERATION_TYPE = 'pca'
TRUNCATED_SVD_OPERATION_TYPE = 'truncated_svd'


class SpectrumNComponentsMethod(str, Enum):
    """Spectrum-based rank selection methods for ``n_components``."""

    ELBOW = 'elbow'
    BROKEN_STICK = 'broken_stick'


SPECTRUM_N_COMPONENTS_METHODS = frozenset(m.value for m in SpectrumNComponentsMethod)
PCA_SUPPORTED_N_COMPONENTS_STR = frozenset({'mle', 'auto'}) | SPECTRUM_N_COMPONENTS_METHODS
TRUNCATED_SVD_SUPPORTED_N_COMPONENTS_STR = frozenset({'auto'}) | SPECTRUM_N_COMPONENTS_METHODS
DECOMPOSITION_MIN_FIT_SAMPLES = 2
AUTO_N_COMPONENTS_ALIASES = frozenset({None, 'auto'})


def is_integral_number(value: Any) -> bool:
    """Return True for integral numbers, excluding ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_real_number(value: Any) -> bool:
    """Return True for real numbers, excluding ``bool``."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_spectrum_n_components_method(value: Any) -> bool:
    """Return True if ``value`` is an elbow / broken-stick method."""
    if isinstance(value, SpectrumNComponentsMethod):
        return True
    return isinstance(value, str) and value in SPECTRUM_N_COMPONENTS_METHODS


def is_auto_n_components(value: Any) -> bool:
    """Return True if ``value`` means the data-dependent ``auto`` budget."""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'auto':
        return True
    return False


def is_valid_pca_n_components(value: Any) -> bool:
    """Return True if ``value`` is a valid PCA ``n_components``."""
    if is_auto_n_components(value):
        return True
    if isinstance(value, SpectrumNComponentsMethod):
        return True
    if isinstance(value, str):
        return value in PCA_SUPPORTED_N_COMPONENTS_STR
    if is_integral_number(value):
        return int(value) >= 1
    if is_real_number(value):
        # Float in (0, 1) is variance ratio; float >= 1 is treated as an int.
        if float(value) < 1.0:
            return float(value) > 0.0
        return True
    return False


def is_valid_truncated_svd_n_components(value: Any) -> bool:
    """Return True if ``value`` is a valid TruncatedSVD ``n_components``."""
    if is_auto_n_components(value):
        return True
    if isinstance(value, SpectrumNComponentsMethod):
        return True
    if isinstance(value, str):
        return value in TRUNCATED_SVD_SUPPORTED_N_COMPONENTS_STR
    if is_integral_number(value):
        return int(value) >= 1
    if is_real_number(value):
        # Non-integral float must be a feature-fraction in (0, 1].
        return 0.0 < float(value) <= 1.0
    return False


def pca_n_components_error_message(value: Any) -> str:
    return (
        f"Unsupported PCA n_components: {value!r}. "
        f"Expected positive int, float in (0, 1], or one of "
        f"{sorted(PCA_SUPPORTED_N_COMPONENTS_STR)}."
    )


def truncated_svd_n_components_error_message(value: Any) -> str:
    return (
        f"Unsupported TruncatedSVD n_components: {value!r}. "
        f"Expected a positive int, float feature-fraction in (0, 1], or one of "
        f"{sorted(TRUNCATED_SVD_SUPPORTED_N_COMPONENTS_STR)}."
    )


def spectrum_method_error_message(value: Any) -> str:
    return (
        f'Unsupported spectrum n_components method: {value!r}. '
        f'Expected one of {sorted(SPECTRUM_N_COMPONENTS_METHODS)}.'
    )


def spectrum_input_error_message(method: Any) -> str:
    name = method.value if isinstance(method, SpectrumNComponentsMethod) else method
    return f'{name} requires singular_values or proportions'


def broken_stick_n_error_message(n: Any) -> str:
    return f'broken_stick_expectations needs n >= 1, got {n}'


def has_enough_decomposition_fit_samples(n_samples: Any) -> bool:
    return is_integral_number(n_samples) and int(n_samples) >= DECOMPOSITION_MIN_FIT_SAMPLES


def decomposition_fit_samples_error_message(n_samples: Any, op_name: str = 'decomposition') -> str:
    return (
        f'{op_name} fit needs at least {DECOMPOSITION_MIN_FIT_SAMPLES} finite samples, '
        f'got {n_samples} after dropping rows with NaN'
    )
