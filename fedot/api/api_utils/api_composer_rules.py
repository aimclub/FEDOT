from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CacheInitPlan:
    use_operations_cache: bool
    use_preprocessing_cache: bool
    use_predictions_cache: bool
    cache_dir: str
    use_stats: bool


@dataclass(frozen=True)
class TunerPlan:
    metric: Any
    iterations: int
    timeout_minutes: float


def build_cache_init_plan(use_operations_cache: bool,
                          use_preprocessing_cache: bool,
                          use_predictions_cache: bool,
                          use_input_preprocessing: bool,
                          cache_dir,
                          use_stats: bool) -> CacheInitPlan:
    return CacheInitPlan(
        use_operations_cache=bool(use_operations_cache),
        use_preprocessing_cache=bool(use_input_preprocessing and use_preprocessing_cache),
        use_predictions_cache=bool(use_predictions_cache),
        cache_dir=cache_dir,
        use_stats=bool(use_stats),
    )


def build_tuner_plan(metrics: Sequence[Any], timeout_minutes: float, iterations: int) -> TunerPlan:
    return TunerPlan(
        metric=metrics[0],
        iterations=iterations,
        timeout_minutes=max(0.0, timeout_minutes),
    )
