from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotValidationError
from fedot.validation.schemas.chunked_ensemble_config import ChunkedEnsembleConfigSchema


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
    min_successful_chunks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_size': self.validation_size,
            'validation_split_seed': self.validation_split_seed,
            'ensemble_method': self.ensemble_method.value,
            'ensemble_params': self.ensemble_params,
            'batch_size': self.batch_size,
            'min_successful_chunks': self.min_successful_chunks,
        }


def validate_chunked_ensemble_config(
    config: Optional[Dict[str, Any]],
    context: Optional[ValidationContext] = None,
) -> ChunkedEnsembleConfig:
    if config is None:
        return ChunkedEnsembleConfig()
    if not isinstance(config, dict):
        raise FedotValidationError(
            '"chunked_ensemble_config" must be a dictionary or None.',
            field_name='_schema',
        )
    loaded = load_validated(
        ChunkedEnsembleConfigSchema(),
        config,
        context,
        prefix='chunked_ensemble_config',
    )
    loaded['ensemble_method'] = EnsembleMethod(loaded['ensemble_method'])
    return ChunkedEnsembleConfig(**loaded)
