from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SamplingStagePlan:
    resolved_predefined_model: Optional[Any]
    should_run_sampling_stage: bool
    skip_metadata: Optional[dict]


@dataclass(frozen=True)
class FinalFitPlan:
    should_train_on_full_dataset: bool


@dataclass(frozen=True)
class ComposerExecutionPlan:
    should_compose: bool
    should_tune: bool
    tuning_timeout_minutes: float


SKIP_REASON_PREDEFINED_MODEL = 'predefined_model'
SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION = 'atomized_initial_assumption'


def is_atomized_initial_assumption(initial_assumption: Optional[Any]) -> bool:
    descriptive_id = getattr(initial_assumption, 'descriptive_id', '')
    return bool(descriptive_id) and 'atomized' in descriptive_id


def plan_sampling_stage(requested_predefined_model: Optional[Any],
                        initial_assumption: Optional[Any],
                        sampling_config_present: bool) -> SamplingStagePlan:
    if requested_predefined_model is not None:
        return SamplingStagePlan(
            resolved_predefined_model=requested_predefined_model,
            should_run_sampling_stage=False,
            skip_metadata=_skip_metadata(SKIP_REASON_PREDEFINED_MODEL) if sampling_config_present else None,
        )

    if is_atomized_initial_assumption(initial_assumption):
        return SamplingStagePlan(
            resolved_predefined_model=initial_assumption,
            should_run_sampling_stage=False,
            skip_metadata=_skip_metadata(SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION) if sampling_config_present else None,
        )

    return SamplingStagePlan(
        resolved_predefined_model=None,
        should_run_sampling_stage=sampling_config_present,
        skip_metadata=None,
    )


def plan_final_fit(history: Optional[Any], pipeline_is_fitted: bool) -> FinalFitPlan:
    return FinalFitPlan(
        should_train_on_full_dataset=history_has_records(history) or not pipeline_is_fitted,
    )


def history_has_records(history: Optional[Any]) -> bool:
    if history is None:
        return False
    is_empty = getattr(history, 'is_empty', None)
    if callable(is_empty):
        return not is_empty()
    return True


def build_composer_execution_plan(with_tuning: bool,
                                  have_time_for_composing: bool,
                                  have_time_for_tuning: bool,
                                  tuning_timeout_minutes: float) -> ComposerExecutionPlan:
    return ComposerExecutionPlan(
        should_compose=have_time_for_composing,
        should_tune=with_tuning and have_time_for_tuning,
        tuning_timeout_minutes=max(0.0, tuning_timeout_minutes),
    )


def _skip_metadata(reason: str) -> dict:
    return {'status': 'skipped', 'reason': reason}
