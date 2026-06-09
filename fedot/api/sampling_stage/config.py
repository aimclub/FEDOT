from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotValidationError
from fedot.validation.schemas.sampling_config import SamplingConfigSchema


@dataclass(frozen=True)
class SamplingConfigBase:
    strategy_kind: Literal['subset', 'chunking']
    provider: str = 'sampling_zoo'
    strategy: str = 'random'
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    cap_max_timeout_share: float = 0.35
    min_automl_time_minutes: float = 0.1
    infinite_timeout_cap_minutes: float = 5.0
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class SamplingSubsetConfig(SamplingConfigBase):
    candidate_ratios: Tuple[float, ...] = (0.15, 0.2, 0.3, 0.5, 0.7)
    delta_metric_threshold: float = 0.03
    validation_size: float = 0.2


@dataclass(frozen=True)
class SamplingChunkingConfig(SamplingConfigBase):
    pass


SamplingConfig = Union[SamplingSubsetConfig, SamplingChunkingConfig]


def validate_sampling_config(
    config: Optional[Dict[str, Any]],
    context: Optional[ValidationContext] = None,
) -> Optional[SamplingConfig]:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise FedotValidationError(
            '"sampling_config" must be a dictionary or None.',
            field_name='_schema',
        )
    loaded = load_validated(
        SamplingConfigSchema(),
        config,
        context,
        prefix='sampling_config',
    )
    if loaded['strategy_kind'] == 'subset':
        return SamplingSubsetConfig(**loaded)
    return SamplingChunkingConfig(**loaded)
