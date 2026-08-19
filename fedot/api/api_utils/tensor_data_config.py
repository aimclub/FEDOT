from dataclasses import fields
from typing import Any, Dict, Optional, Set

from fedot.api.api_utils.schemas import TensorDataConfigSchema
from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotValidationError, unknown_keys_error

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


def validate_tensor_data_config(
    config: Optional[Dict[str, Any]],
    context: Optional[ValidationContext] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate user-provided ``tensor_data_config`` for :class:`~fedot.api.api_utils.params.ApiParams`.

    The config is a flat dictionary of options forwarded to
    :meth:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator.create`
    (as ``DataSpec`` kwargs) plus ``backend_name``. Runtime values such as ``task``,
    ``state``, and ``target`` must not be set here — they are injected when data
    is created during ``fit`` / ``predict``.

    Args:
        config: User config dictionary or ``None``.
        context: Optional validation context for default-or-raise recovery.

    Returns:
        A shallow copy of the validated config, or ``None`` when *config* is ``None``.

    Raises:
        FedotValidationError: If *config* is not a dict, contains unknown / forbidden
            keys, or has an invalid ``use_cache`` / ``backend_name``.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise FedotValidationError(
            '"tensor_data_config" must be a dictionary or None.',
            field_name='_schema',
        )

    unknown_keys = set(config) - _ALLOWED_KEYS
    if unknown_keys:
        raise unknown_keys_error('tensor_data_config', unknown_keys)

    forbidden_keys = set(config) & _RUNTIME_KEYS
    if forbidden_keys:
        raise FedotValidationError(
            'Keys reserved for runtime injection must not appear in '
            f'"tensor_data_config": {sorted(forbidden_keys)}',
            field_name='_schema',
        )

    normalized = load_validated(
        TensorDataConfigSchema(),
        dict(config),
        context,
        prefix='tensor_data_config',
    )
    if 'backend_name' in normalized:
        normalized['backend_name'] = Backend.normalize_name(normalized['backend_name'])

    return normalized


def resolve_tensor_data_config(
    user_config: Optional[Dict[str, Any]],
    *,
    use_preprocessing_cache: bool = True,
) -> Dict[str, Any]:
    """
    Build the validated TensorDataCreator config stored on :class:`~fedot.api.api_utils.params.ApiParams`.

    User options from ``tensor_data_config`` are validated first. Missing ``backend_name``
    defaults to ``'cpu'``. ``use_cache`` may be set here for :class:`~fedot.core.data.tensor_data.data_spec.DataSpec`
    (standalone / ``create_data``); :class:`~fedot.api.main.Fedot` also applies
    ``use_cache`` on init and when calling :meth:`~fedot.api.main.Fedot.create_data`.
    ``use_preprocessing_cache`` is unused here.
    """
    validated = validate_tensor_data_config(user_config) or {}
    config = dict(validated)
    config.setdefault('backend_name', Backend.DEFAULT_NAME)
    return validate_tensor_data_config(config)
