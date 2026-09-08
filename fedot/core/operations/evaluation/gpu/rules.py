from dataclasses import dataclass
from typing import Any, Mapping


TENSOR_UNSUPPORTED_CUML_OPERATIONS = frozenset({'sgd'})
HIGH_PRECISION_CUML_OPERATIONS = frozenset({'logit'})


@dataclass(frozen=True)
class CuMLRuntimePlan:
    """Data boundary contract for one cuML fit or predict call."""

    returns_tensor: bool
    result_device: str


@dataclass(frozen=True)
class CuMLPrecisionPlan:
    """Numeric precision required for stable training of an operation."""

    dtype_name: str


def build_cuml_runtime_plan(is_tensor_data: bool, feature_device: str = 'cpu') -> CuMLRuntimePlan:
    """Keep TensorData results on their source device and legacy results as NumPy."""
    return CuMLRuntimePlan(
        returns_tensor=is_tensor_data,
        result_device=feature_device if is_tensor_data else 'cpu',
    )


def build_cuml_precision_plan(operation_type: str) -> CuMLPrecisionPlan:
    dtype_name = 'float64' if operation_type in HIGH_PRECISION_CUML_OPERATIONS else 'float32'
    return CuMLPrecisionPlan(dtype_name=dtype_name)


def adapt_cuml_parameters(operation_type: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate shared FEDOT/sklearn parameters to the native cuML API."""
    adapted = dict(parameters)
    adapted.pop('n_jobs', None)

    if operation_type in {'rf', 'rfr'} and 'criterion' in adapted:
        adapted['split_criterion'] = adapted.pop('criterion')
    if operation_type in {'rf', 'rfr'}:
        adapted.setdefault('n_bins', 256)

    if operation_type == 'logit':
        adapted.setdefault('max_iter', 10000)
        adapted.setdefault('tol', 1e-7)
        adapted.setdefault('linesearch_max_iter', 200)

    if operation_type == 'svc':
        adapted['probability'] = True

    if operation_type == 'kmeans':
        adapted.setdefault('n_clusters', 2)

    adapted['output_type'] = 'cupy'
    return adapted


def validate_cuml_tensor_operation(operation_type: str):
    """Reject historical solver aliases that do not implement a model contract."""
    if operation_type in TENSOR_UNSUPPORTED_CUML_OPERATIONS:
        raise ValueError(
            f'cuML operation {operation_type!r} is a low-level solver and has no TensorData model contract'
        )
