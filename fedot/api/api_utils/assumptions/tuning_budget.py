"""Bound cross-validation trials without disabling hyperparameter tuning."""

from math import isfinite
from numbers import Real
from typing import Optional, Tuple

from fedot.core.constants import MINIMAL_SECONDS_FOR_TUNING


def selected_pipeline_fold_seconds(history, pipeline, cv_folds) -> Optional[float]:
    """Read measured CV wall time for this exact graph, in seconds.

    GOLEM individual metadata includes fit, prediction and preprocessing for
    the entire CV. Pipeline.computation_time is a different, minute-valued fit
    timer and must not be used here. Ignore invalid and unrelated individuals.
    """
    if history is None or not cv_folds or cv_folds < 2:
        return None
    generations = getattr(history, 'generations', None)
    if not isinstance(generations, (list, tuple)):
        return None
    durations = []
    for generation in generations:
        for individual in generation:
            if not individual.fitness.valid or individual.graph.descriptive_id != pipeline.descriptive_id:
                continue
            seconds = individual.metadata.get('computation_time_in_seconds')
            if isinstance(seconds, Real) and not isinstance(seconds, bool) and isfinite(seconds) and seconds > 0:
                durations.append(float(seconds) / cv_folds)
    return max(durations) if durations else None


def measured_tuning_resources(available_seconds: float, fold_seconds: float,
                              cv_folds: int) -> Optional[Tuple[float, float, int]]:
    """Budget initial CV, at least one trial, final CV, with 30% headroom.

    The caller reserves full-data refit separately. None means *skip tuning*,
    not fall back to an unbounded tuner. Keep at least two validation folds.
    """
    for folds in range(min(cv_folds, 3), 1, -1):
        evaluation = 1.3 * fold_seconds * folds
        if 3 * evaluation + MINIMAL_SECONDS_FOR_TUNING <= available_seconds:
            # Reserve an in-flight trial overrun and the final CV check.
            return available_seconds - 2 * evaluation, evaluation, folds
    return None


def bounded_composition_resources(total_seconds: float, composing_seconds: float,
                                  estimated_cv_seconds: float) -> Optional[Tuple[float, float]]:
    """Reserve useful tuning time and bound an in-flight evolutionary CV trial.

    GOLEM checks its composition deadline between evaluations. On an expensive
    task, the last five-fold evaluation can overrun that deadline by minutes.
    Small tasks keep the existing 60/40 composing/tuning split.
    """
    if (total_seconds <= 0 or composing_seconds <= 0 or estimated_cv_seconds < 120 or
            estimated_cv_seconds >= 0.3 * composing_seconds):
        return None

    composing_seconds = min(composing_seconds, 0.4 * total_seconds)
    # Allow headroom for slower candidates, but never let one evaluation
    # consume the remaining tuning allocation.
    evaluation_seconds = min(0.6 * composing_seconds,
                             max(2 * estimated_cv_seconds, 0.5 * composing_seconds))
    return composing_seconds, evaluation_seconds


def bounded_tuning_resources(available_seconds: float, initial_fold_seconds: float,
                             cv_folds: Optional[int],
                             complex_pipeline: bool = False) -> Optional[Tuple[float, float, int]]:
    """Return search timeout, per-CV-evaluation limit and number of folds.

    GOLEM's Hyperopt timeout is checked *between* trials; an in-flight CV
    evaluation and the final CV check can each overrun it. For expensive fits,
    reserve both evaluations and shorten the search deadline accordingly.
    Small problems retain their original tuning configuration unless evolution
    produced a complex graph whose tuning cost cannot be inferred from the
    initial fit.
    """
    # A fast initial assumption does not predict the cost of tuning an evolved
    # multi-node graph. Its search can contain several expensive learners even
    # when the baseline fit took only a few seconds.
    if (available_seconds <= 0 or initial_fold_seconds <= 0 or
            cv_folds is None or cv_folds < 2 or
            (initial_fold_seconds * cv_folds < 45 and not complex_pipeline)):
        return None

    folds = min(cv_folds, 3)
    if folds > 2 and initial_fold_seconds * folds > 0.3 * available_seconds:
        folds = 2
    initial_cv_seconds = initial_fold_seconds * folds
    evaluation_seconds = min(0.3 * available_seconds,
                             max(1.3 * initial_cv_seconds, 0.25 * available_seconds))
    search_seconds = available_seconds - 2 * evaluation_seconds
    if initial_cv_seconds + MINIMAL_SECONDS_FOR_TUNING >= search_seconds:
        return None
    return search_seconds, evaluation_seconds, folds
