from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig, validate_chunked_ensemble_config
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class SamplingStagePlan:
    should_run_sampling_stage: bool
    skip_metadata: Optional[dict]


@dataclass(frozen=True)
class ChunkedEnsemblePlan:
    should_use_chunked_ensemble: bool
    config: Optional[ChunkedEnsembleConfig]
    train_split_ratio: float
    should_select_class_representatives: bool
    validation_split_seed: Optional[int]

    def require_config(self) -> ChunkedEnsembleConfig:
        if self.config is None:
            raise ValueError('Chunked ensemble config is required by this plan.')
        return self.config


@dataclass(frozen=True)
class FinalFitPlan:
    action: 'FinalFitAction'


class FinalFitAction(Enum):
    fit_pipeline_on_full_data = 'fit_pipeline_on_full_data'
    finalize_ensemble = 'finalize_ensemble'
    skip = 'skip'


@dataclass(frozen=True)
class ComposerExecutionPlan:
    should_compose: bool
    should_tune: bool
    tuning_timeout_minutes: float


SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION = 'atomized_initial_assumption'


def is_atomized_initial_assumption(initial_assumption: Optional[Any]) -> bool:
    descriptive_id = getattr(initial_assumption, 'descriptive_id', '')
    return bool(descriptive_id) and 'atomized' in descriptive_id


def plan_sampling_stage(initial_assumption: Optional[Any],
                        sampling_config_present: bool) -> SamplingStagePlan:
    if is_atomized_initial_assumption(initial_assumption):
        return SamplingStagePlan(
            should_run_sampling_stage=False,
            skip_metadata=_skip_metadata(SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION) if sampling_config_present else None,
        )

    return SamplingStagePlan(
        should_run_sampling_stage=sampling_config_present,
        skip_metadata=None,
    )


def plan_chunked_ensemble(should_run_sampling_stage: bool,
                          strategy_kind: Optional[str],
                          task_type: Any,
                          chunked_ensemble_config: Optional[dict] = None) -> ChunkedEnsemblePlan:
    supported_task_types = (TaskTypesEnum.classification, TaskTypesEnum.regression)
    should_use_chunked_ensemble = (
        should_run_sampling_stage
        and strategy_kind == 'chunking'
        and task_type in supported_task_types
    )
    config = validate_chunked_ensemble_config(chunked_ensemble_config) if should_use_chunked_ensemble else None
    validation_size = config.validation_size if config is not None else ChunkedEnsembleConfig().validation_size
    validation_split_seed = config.validation_split_seed if should_use_chunked_ensemble else None
    return ChunkedEnsemblePlan(
        should_use_chunked_ensemble=should_use_chunked_ensemble,
        config=config,
        train_split_ratio=1.0 - validation_size,
        should_select_class_representatives=should_use_chunked_ensemble and task_type is TaskTypesEnum.classification,
        validation_split_seed=validation_split_seed,
    )


def plan_final_fit(history: Optional[Any],
                   pipeline_is_fitted: bool,
                   is_pipeline_ensemble: bool = False) -> FinalFitPlan:
    if is_pipeline_ensemble:
        return FinalFitPlan(action=FinalFitAction.finalize_ensemble)

    if history_has_records(history) or not pipeline_is_fitted:
        return FinalFitPlan(action=FinalFitAction.fit_pipeline_on_full_data)

    return FinalFitPlan(
        action=FinalFitAction.skip,
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
