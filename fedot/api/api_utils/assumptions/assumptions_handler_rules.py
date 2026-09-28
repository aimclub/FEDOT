from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

from fedot.api.time import ApiTime
from fedot.core.pipelines.pipeline import Pipeline


@dataclass(frozen=True)
class AssumptionFitError:
    code: str
    message: str
    cause: str
    exception: Optional[Exception] = None


@dataclass(frozen=True)
class PresetDecision:
    preset: str
    was_changed: bool


NormalizedInitialAssumption = Optional[List[Pipeline]]


def normalize_initial_assumption(
        initial_assumption: Union[List[Pipeline], Pipeline, None]) -> NormalizedInitialAssumption:
    if initial_assumption is None:
        return None
    if isinstance(initial_assumption, Pipeline):
        return [initial_assumption]
    return initial_assumption


def resolve_initial_assumption(initial_assumption: Union[List[Pipeline], Pipeline, None],
                               builder: Callable[[], List[Pipeline]]) -> List[Pipeline]:
    normalized = normalize_initial_assumption(initial_assumption)
    if normalized is None:
        return builder()
    return normalized


def build_assumption_fit_error(ex: Exception) -> AssumptionFitError:
    fit_failed_info = f'Initial pipeline fit was failed due to: {ex}.'
    advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'
    return AssumptionFitError(
        code='initial_assumption_fit_failed',
        message=advice_info,
        cause=str(ex),
        exception=ex,
    )


def decide_preset(preset: Optional[str],
                  timer: ApiTime,
                  n_jobs: int,
                  chooser: Callable[[ApiTime, int], str]) -> PresetDecision:
    if not preset or preset == 'auto':
        return PresetDecision(
            preset=chooser(timer, n_jobs),
            was_changed=True,
        )
    return PresetDecision(preset=preset, was_changed=False)
