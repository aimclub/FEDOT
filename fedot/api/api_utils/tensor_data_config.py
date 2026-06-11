from dataclasses import fields
from typing import Any, Dict, Optional, Set

from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data.data_spec import DataSpec

_CREATOR_ONLY_KEYS: Set[str] = {'backend_name'}

# Injected at fit/predict time or filled during TensorDataCreator pipeline.
_RUNTIME_KEYS: Set[str] = {
    'task',
    'state',
    'target',
    'features',
    'idx',
    'idx_mapping',
    'ts_init_shape',
    'predict',
}

_USER_CONFIGURABLE_DATA_SPEC_KEYS: Set[str] = {
    field.name for field in fields(DataSpec)
} - _RUNTIME_KEYS

_ALLOWED_KEYS: Set[str] = _USER_CONFIGURABLE_DATA_SPEC_KEYS | _CREATOR_ONLY_KEYS

def validate_tensor_data_config(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validate user-provided ``tensor_data_config`` for :class:`~fedot.api.api_utils.params.ApiParams`.

    The config is a flat dictionary of options forwarded to
    :meth:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator.create`
    (as ``DataSpec`` kwargs) plus ``backend_name``. Runtime values such as ``task``,
    ``state``, and ``target`` must not be set here — they are injected when data
    is created during ``fit`` / ``predict``.

    Args:
        config: User config dictionary or ``None``.

    Returns:
        A shallow copy of the validated config, or ``None`` when *config* is ``None``.

    Raises:
        ValueError: If *config* is not a dict or contains unknown / forbidden keys.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('"tensor_data_config" must be a dictionary or None.')

    unknown_keys = set(config) - _ALLOWED_KEYS
    if unknown_keys:
        raise ValueError(
            f'Unknown keys in "tensor_data_config": {sorted(unknown_keys)}'
        )

    forbidden_keys = set(config) & _RUNTIME_KEYS
    if forbidden_keys:
        raise ValueError(
            'Keys reserved for runtime injection must not appear in '
            f'"tensor_data_config": {sorted(forbidden_keys)}'
        )

    # TODO romankuklo: should remove from here?
    normalized = dict(config)
    if 'backend_name' in normalized:
        normalized['backend_name'] = Backend.normalize_name(normalized['backend_name'])

    if 'use_cache' in normalized and not isinstance(normalized['use_cache'], bool):
        raise ValueError('"tensor_data_config.use_cache" must be a boolean.')

    return normalized


def resolve_tensor_data_config(
    user_config: Optional[Dict[str, Any]],
    *,
    use_preprocessing_cache: bool = True,
) -> Dict[str, Any]:
    """
    Build the validated TensorDataCreator config stored on :class:`~fedot.api.api_utils.params.ApiParams`.

    User options from ``tensor_data_config`` are validated first. Missing ``backend_name``
    defaults to ``'cpu'``; missing ``use_cache`` follows ``use_preprocessing_cache``.
    """
    validated = validate_tensor_data_config(user_config) or {}
    config = dict(validated)
    config.setdefault('backend_name', Backend.DEFAULT_NAME)
    if 'use_cache' not in config:
        config['use_cache'] = use_preprocessing_cache
    return validate_tensor_data_config(config)
