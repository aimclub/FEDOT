from typing import Any, Dict, Optional, Tuple

from fedot.core.backend.backend import Backend
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator


def _resolve_target_argument(target: Any, target_idx: Any) -> Tuple[Any, Any]:
    """Map public ``target`` to creator ``target`` / ``target_idx``."""
    if isinstance(target, str):
        if target_idx is not None:
            raise ValueError(
                'Pass either target=<column name> or target_idx=..., not both.'
            )
        return None, target
    return target, target_idx


def _resolve_from_data(
    from_data: Optional[TensorData],
    *,
    task: Any,
    data_type: Any,
    spec_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill predict-time defaults from a previously created ``TensorData``."""
    resolved = dict(spec_kwargs)

    if from_data is None:
        if task is not None:
            resolved['task'] = task
        if data_type is not None:
            resolved['data_type'] = data_type
        return resolved

    if not isinstance(from_data, TensorData):
        raise TypeError(
            f'from_data must be TensorData, got {type(from_data).__name__}.'
        )

    resolved.setdefault('state', StateEnum.PREDICT)
    resolved['task'] = task if task is not None else from_data.task
    resolved['data_type'] = data_type if data_type is not None else from_data.data_type
    if 'trace_uuid' not in resolved:
        resolved['trace_uuid'] = from_data.trace_uuid
    return resolved


def _build_create_data_kwargs(
    *,
    target: Any = None,
    target_idx: Any = None,
    backend: str = Backend.DEFAULT_NAME,
    task: Any = None,
    data_type: Any = None,
    from_data: Optional[TensorData] = None,
    **options: Any,
) -> Tuple[str, Dict[str, Any]]:
    resolved_target, resolved_target_idx = _resolve_target_argument(target, target_idx)
    spec_kwargs = _resolve_from_data(
        from_data,
        task=task,
        data_type=data_type,
        spec_kwargs=dict(options),
    )
    if resolved_target is not None:
        spec_kwargs['target'] = resolved_target
    if resolved_target_idx is not None:
        spec_kwargs['target_idx'] = resolved_target_idx

    return Backend.normalize_name(backend), spec_kwargs


def create_data(
    features: Any,
    target: Any = None,
    *,
    backend: str = Backend.DEFAULT_NAME,
    task: Any = None,
    data_type: Any = None,
    from_data: Optional[TensorData] = None,
    target_idx: Any = None,
    **options: Any,
) -> TensorData:
    """
    Create :class:`~fedot.core.data.tensor_data.tensor_data.TensorData` for Fedot.

    This is the public entrypoint for turning raw sources (numpy / pandas / CSV /
    paths) into the internal ``TensorData`` type used by :class:`~fedot.api.main.Fedot`.

    Args:
        features: Features matrix, dataframe, or path supported by the data readers.
        target: Target values, or a column name (``str``) to extract from ``features``.
        backend: Compute backend, ``'cpu'`` (default) or ``'gpu'``.
        task: FEDOT task or task name. Defaults to classification when omitted.
        data_type: Data type or alias (e.g. ``'tabular'``, ``'time_series'``).
        from_data: Previously created train :class:`TensorData`. When set, predict
            mode is used and ``task``, ``data_type``, and ``trace_uuid`` are taken
            from it (unless overridden explicitly).
        target_idx: Target column index/name when not using ``target=<column name>``.
        **options: Extra :class:`~fedot.core.data.tensor_data.data_spec.DataSpec`
            options (``encoding_strategy``, ``ts_orientation``, ``use_cache``, …).

    Returns:
        Prepared :class:`TensorData` on the requested backend.

    Examples:
        >>> train = create_data(X_train, target=y_train)
        >>> test = create_data(X_test, from_data=train)
        >>> train = create_data('train.csv', target='target')
    """
    backend_name, spec_kwargs = _build_create_data_kwargs(
        target=target,
        target_idx=target_idx,
        backend=backend,
        task=task,
        data_type=data_type,
        from_data=from_data,
        **options,
    )
    return TensorDataCreator.create(features, backend_name, **spec_kwargs)


def create_data_lazy(
    features: Any,
    target: Any = None,
    *,
    backend: str = Backend.DEFAULT_NAME,
    task: Any = None,
    data_type: Any = None,
    from_data: Optional[TensorData] = None,
    target_idx: Any = None,
    **options: Any,
):
    """
    Lazy variant of :func:`create_data`.

    Returns a :class:`~fedot.core.data.tensor_data.lazy_tensor.LazyTensor` that
    materializes ``TensorData`` on first access.
    """
    backend_name, spec_kwargs = _build_create_data_kwargs(
        target=target,
        target_idx=target_idx,
        backend=backend,
        task=task,
        data_type=data_type,
        from_data=from_data,
        **options,
    )
    return TensorDataCreator.create_lazy(features, backend_name, **spec_kwargs)
