from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Optional


class EnsembleMethod(str, Enum):
    voting = 'voting'
    weighted = 'weighted'
    routed_weighted = 'routed_weighted'
    gated_weighted = 'gated_weighted'


@dataclass(frozen=True)
class ChunkedEnsembleConfig:
    validation_size: float = 0.2
    validation_split_seed: Optional[int] = 42
    ensemble_method: EnsembleMethod = EnsembleMethod.voting
    ensemble_params: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_size': self.validation_size,
            'validation_split_seed': self.validation_split_seed,
            'ensemble_method': self.ensemble_method.value,
            'ensemble_params': self.ensemble_params,
            'batch_size': self.batch_size,
        }


def validate_chunked_ensemble_config(config: Optional[Dict[str, Any]]) -> ChunkedEnsembleConfig:
    if config is None:
        return ChunkedEnsembleConfig()
    if not isinstance(config, dict):
        raise ValueError('"chunked_ensemble_config" must be a dictionary or None.')

    allowed_keys = {field.name for field in fields(ChunkedEnsembleConfig)}
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(f'Unknown keys in "chunked_ensemble_config": {sorted(unknown_keys)}')

    normalized_config = dict(config)
    if 'ensemble_method' in normalized_config:
        normalized_config['ensemble_method'] = _validate_ensemble_method(normalized_config['ensemble_method'])

    merged = ChunkedEnsembleConfig(**normalized_config)
    _validate_chunked_ensemble_config_values(merged)
    return merged


def _validate_chunked_ensemble_config_values(config: ChunkedEnsembleConfig) -> None:
    if not 0 < config.validation_size < 1:
        raise ValueError('"chunked_ensemble_config.validation_size" must be in range (0, 1).')
    if config.validation_split_seed is not None and not isinstance(config.validation_split_seed, int):
        raise ValueError('"chunked_ensemble_config.validation_split_seed" must be int or None.')
    if not isinstance(config.ensemble_method, EnsembleMethod):
        raise ValueError(
            '"chunked_ensemble_config.ensemble_method" must be one of '
            '{"voting", "weighted", "routed_weighted", "gated_weighted"}.'
        )
    if not isinstance(config.ensemble_params, dict):
        raise ValueError('"chunked_ensemble_config.ensemble_params" must be a dictionary.')
    if config.batch_size <= 0:
        raise ValueError('"chunked_ensemble_config.batch_size" must be > 0.')


def _validate_ensemble_method(value: Any) -> EnsembleMethod:
    try:
        return EnsembleMethod(value)
    except ValueError as ex:
        raise ValueError(
            '"chunked_ensemble_config.ensemble_method" must be one of '
            '{"voting", "weighted", "routed_weighted", "gated_weighted"}.'
        ) from ex
