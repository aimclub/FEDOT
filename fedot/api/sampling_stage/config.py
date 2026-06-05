from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union


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

_CONFIG_BY_STRATEGY_KIND = {
    'subset': SamplingSubsetConfig,
    'chunking': SamplingChunkingConfig,
}


def validate_sampling_config(config: Optional[Dict[str, Any]]) -> Optional[SamplingConfig]:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('"sampling_config" must be a dictionary or None.')

    strategy_kind = config.get('strategy_kind')
    if strategy_kind is None:
        raise ValueError('"sampling_config.strategy_kind" must be provided.')
    if strategy_kind not in ('subset', 'chunking'):
        raise ValueError('"sampling_config.strategy_kind" must be "subset" or "chunking".')

    config_cls = _CONFIG_BY_STRATEGY_KIND[strategy_kind]
    allowed_keys = _field_names(config_cls)
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f'Unknown keys in "sampling_config": {sorted(unknown_keys)}')

    if strategy_kind == 'subset':
        merged = SamplingSubsetConfig(**config)
        _validate_base_config_values(merged)
        _validate_subset_config_values(merged)
        return merged

    merged = SamplingChunkingConfig(**config)
    _validate_base_config_values(merged)
    _validate_chunking_config_values(merged)
    return merged


def _field_names(config_cls: Any) -> set:
    return {field.name for field in fields(config_cls)}


def _validate_base_config_values(config: SamplingConfigBase) -> None:
    if not isinstance(config.provider, str) or not config.provider.strip():
        raise ValueError(
            '"sampling_config.provider" must be a non-empty string.')

    if not isinstance(config.strategy, str) or not config.strategy.strip():
        raise ValueError(
            '"sampling_config.strategy" must be a non-empty string.')

    if not isinstance(config.strategy_params, dict):
        raise ValueError(
            '"sampling_config.strategy_params" must be a dictionary.')

    if not 0 < config.cap_max_timeout_share <= 1:
        raise ValueError(
            '"sampling_config.cap_max_timeout_share" must be in range (0, 1].')

    if config.min_automl_time_minutes <= 0:
        raise ValueError(
            '"sampling_config.min_automl_time_minutes" must be > 0.')

    if config.infinite_timeout_cap_minutes <= 0:
        raise ValueError(
            '"sampling_config.infinite_timeout_cap_minutes" must be > 0.')

    if config.random_state is not None and not isinstance(config.random_state, int):
        raise ValueError('"sampling_config.random_state" must be int or None.')


def _validate_subset_config_values(config: SamplingSubsetConfig) -> None:
    ratios = _validate_ratios(config.candidate_ratios)
    if ratios != tuple(config.candidate_ratios):
        raise ValueError('"sampling_config.candidate_ratios" must be sorted in ascending order without duplicates.')

    if config.delta_metric_threshold < 0:
        raise ValueError('"sampling_config.delta_metric_threshold" must be >= 0.')

    if not 0 < config.validation_size < 1:
        raise ValueError('"sampling_config.validation_size" must be in range (0, 1).')


def _validate_chunking_config_values(config: SamplingChunkingConfig) -> None:
    pass


def _validate_ratios(ratios: Sequence[float]) -> Tuple[float, ...]:
    if not isinstance(ratios, (list, tuple)) or len(ratios) == 0:
        raise ValueError(
            '"sampling_config.candidate_ratios" must be a non-empty list of floats.')

    normalized = []
    for ratio in ratios:
        if not isinstance(ratio, (float, int)):
            raise ValueError(
                '"sampling_config.candidate_ratios" must contain only numbers.')
        ratio = float(ratio)
        if not 0 < ratio <= 1:
            raise ValueError(
                '"sampling_config.candidate_ratios" values must be in range (0, 1].')
        normalized.append(ratio)

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            '"sampling_config.candidate_ratios" must not contain duplicates.')

    sorted_ratios = sorted(normalized)
    return tuple(sorted_ratios)
