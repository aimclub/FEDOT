from typing import Any, Dict, Optional, Tuple, Union

from pymonad.either import Left, Right

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.extensions.contracts import ExtensionError, ExternalModelSpec


RuntimeReservedKeys = ('model_fit', 'model_predict')


def normalize_extension_user_params(user_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return dict(user_params or {})


def apply_extension_defaults(defaults: Dict[str, Any], user_params: Dict[str, Any]) -> Dict[str, Any]:
    return {**defaults, **user_params}


def find_missing_required_params(required: Tuple[str, ...], params: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(param_name for param_name in required if param_name not in params)


def resolve_extension_params(model_spec: ExternalModelSpec,
                             user_params: Optional[Dict[str, Any]] = None):
    normalized_user_params = normalize_extension_user_params(user_params)
    resolved_params = apply_extension_defaults(model_spec.hyperparams_schema.defaults, normalized_user_params)
    missing_required_params = find_missing_required_params(model_spec.hyperparams_schema.required, resolved_params)

    if missing_required_params:
        return Left(ExtensionError(
            code='missing_required_hyperparams',
            message=f'Extension model "{model_spec.name}" is missing required hyperparameters.',
            details={'required': list(missing_required_params)},
        ))

    return Right(resolved_params)


def extract_factory_params(strategy_params: Union[Dict[str, Any], OperationParameters]) -> Dict[str, Any]:
    if isinstance(strategy_params, OperationParameters):
        strategy_params = strategy_params.to_dict()
    return {
        key: value
        for key, value in strategy_params.items()
        if not key.startswith('_') and key not in RuntimeReservedKeys
    }
