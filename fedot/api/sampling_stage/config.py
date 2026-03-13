from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class SamplingConfigBase:
    strategy_kind: Literal['subset', 'chunking']
    provider: str = 'sampling_zoo'
    strategy: str = 'random'
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    budget_policy: str = 'dynamic_cap'
    cap_max_timeout_share: float = 0.35
    min_automl_time_minutes: float = 0.1
    infinite_timeout_cap_minutes: float = 5.0
    error_policy: str = 'fail_fast'
    artifact_mode: str = 'minimal'
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class SamplingSubsetConfig(SamplingConfigBase):
    candidate_ratios: Tuple[float, ...] = (0.15, 0.2, 0.3, 0.5, 0.7)
    delta_metric_threshold: float = 0.03
    delta_type: str = 'relative'
    validation_size: float = 0.2
    guard_max_rank: int = 256
    guard_max_modes: int = 4
    guard_max_sample_size: int = 100000


@dataclass(frozen=True)
class SamplingChunkingConfig(SamplingConfigBase):
    guard_max_partitions: int = 128


SamplingConfig = Union[SamplingSubsetConfig, SamplingChunkingConfig]

_BASE_KEYS = {
    'strategy_kind',
    'provider',
    'strategy',
    'strategy_params',
    'budget_policy',
    'cap_max_timeout_share',
    'min_automl_time_minutes',
    'infinite_timeout_cap_minutes',
    'error_policy',
    'artifact_mode',
    'random_state',
}

_SUBSET_KEYS = {
    'candidate_ratios',
    'delta_metric_threshold',
    'delta_type',
    'validation_size',
    'guard_max_rank',
    'guard_max_modes',
    'guard_max_sample_size',
}

_CHUNKING_KEYS = {
    'guard_max_partitions',
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

    allowed_keys = _BASE_KEYS | (_SUBSET_KEYS if strategy_kind == 'subset' else _CHUNKING_KEYS)
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(f'Unknown keys in "sampling_config": {sorted(unknown_keys)}')

    if strategy_kind == 'subset':
        merged = SamplingSubsetConfig(**config)
        _validate_base_config_values(merged)
        _validate_subset_config_values(merged)
        return merged

    merged = SamplingChunkingConfig(**config)
    _validate_base_config_values(merged)
    _validate_chunking_config_values(merged)
    return merged


def _validate_base_config_values(config: SamplingConfigBase) -> None:
    if not isinstance(config.provider, str) or not config.provider.strip():
        raise ValueError('"sampling_config.provider" must be a non-empty string.')

    if not isinstance(config.strategy, str) or not config.strategy.strip():
        raise ValueError('"sampling_config.strategy" must be a non-empty string.')

    if not isinstance(config.strategy_params, dict):
        raise ValueError('"sampling_config.strategy_params" must be a dictionary.')

    if config.budget_policy != 'dynamic_cap':
        raise ValueError('"sampling_config.budget_policy" supports only "dynamic_cap" in V1.')

    if not 0 < config.cap_max_timeout_share <= 1:
        raise ValueError('"sampling_config.cap_max_timeout_share" must be in range (0, 1].')

    if config.min_automl_time_minutes <= 0:
        raise ValueError('"sampling_config.min_automl_time_minutes" must be > 0.')

    if config.infinite_timeout_cap_minutes <= 0:
        raise ValueError('"sampling_config.infinite_timeout_cap_minutes" must be > 0.')

    if config.error_policy != 'fail_fast':
        raise ValueError('"sampling_config.error_policy" supports only "fail_fast" in V1.')

    if config.artifact_mode != 'minimal':
        raise ValueError('"sampling_config.artifact_mode" supports only "minimal" in V1.')

    if config.random_state is not None and not isinstance(config.random_state, int):
        raise ValueError('"sampling_config.random_state" must be int or None.')


def _validate_subset_config_values(config: SamplingSubsetConfig) -> None:
    ratios = _validate_ratios(config.candidate_ratios)
    if ratios != tuple(config.candidate_ratios):
        raise ValueError('"sampling_config.candidate_ratios" must be sorted in ascending order without duplicates.')

    if config.delta_metric_threshold < 0:
        raise ValueError('"sampling_config.delta_metric_threshold" must be >= 0.')

    if config.delta_type not in {'relative', 'absolute'}:
        raise ValueError('"sampling_config.delta_type" must be one of {"relative", "absolute"}.')

    if not 0 < config.validation_size < 1:
        raise ValueError('"sampling_config.validation_size" must be in range (0, 1).')

    for key in ('guard_max_rank', 'guard_max_modes', 'guard_max_sample_size'):
        if getattr(config, key) <= 0:
            raise ValueError(f'"sampling_config.{key}" must be > 0.')

    _validate_subset_strategy_param_guards(config)


def _validate_chunking_config_values(config: SamplingChunkingConfig) -> None:
    if config.guard_max_partitions <= 0:
        raise ValueError('"sampling_config.guard_max_partitions" must be > 0.')
    _validate_chunking_strategy_param_guards(config)


def _validate_ratios(ratios: Sequence[float]) -> Tuple[float, ...]:
    if not isinstance(ratios, (list, tuple)) or len(ratios) == 0:
        raise ValueError('"sampling_config.candidate_ratios" must be a non-empty list of floats.')

    normalized = []
    for ratio in ratios:
        if not isinstance(ratio, (float, int)):
            raise ValueError('"sampling_config.candidate_ratios" must contain only numbers.')
        ratio = float(ratio)
        if not 0 < ratio <= 1:
            raise ValueError('"sampling_config.candidate_ratios" values must be in range (0, 1].')
        normalized.append(ratio)

    if len(set(normalized)) != len(normalized):
        raise ValueError('"sampling_config.candidate_ratios" must not contain duplicates.')

    sorted_ratios = sorted(normalized)
    return tuple(sorted_ratios)


def _validate_subset_strategy_param_guards(config: SamplingSubsetConfig) -> None:
    params = config.strategy_params

    for rank_key in ('rank', 'approx_rank'):
        if rank_key not in params:
            continue
        rank = params[rank_key]
        if isinstance(rank, (list, tuple)):
            if any(float(r) > config.guard_max_rank for r in rank if isinstance(r, (int, float)) and float(r) > 1):
                raise ValueError(
                    f'"sampling_config.strategy_params.{rank_key}" exceeds guard_max_rank={config.guard_max_rank}.'
                )
        elif isinstance(rank, (int, float)) and float(rank) > 1 and float(rank) > config.guard_max_rank:
            raise ValueError(
                f'"sampling_config.strategy_params.{rank_key}" exceeds guard_max_rank={config.guard_max_rank}.'
            )

    modes = params.get('modes')
    if modes is not None:
        if not isinstance(modes, (list, tuple)):
            raise ValueError('"sampling_config.strategy_params.modes" must be a list/tuple.')
        if len(modes) > config.guard_max_modes:
            raise ValueError(
                f'"sampling_config.strategy_params.modes" exceeds guard_max_modes={config.guard_max_modes}.'
            )

    sample_size = params.get('sample_size')
    if sample_size is not None and isinstance(sample_size, int) and sample_size > config.guard_max_sample_size:
        raise ValueError(
            f'"sampling_config.strategy_params.sample_size" exceeds guard_max_sample_size='
            f'{config.guard_max_sample_size}.'
        )


def _validate_chunking_strategy_param_guards(config: SamplingChunkingConfig) -> None:
    params = config.strategy_params
    for key in ('n_partitions', 'partitions', 'n_splits'):
        value = params.get(key)
        if value is not None and isinstance(value, int) and value > config.guard_max_partitions:
            raise ValueError(
                f'"sampling_config.strategy_params.{key}" exceeds guard_max_partitions={config.guard_max_partitions}.'
            )
