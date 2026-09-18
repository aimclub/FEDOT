from types import SimpleNamespace

from fedot.api.api_utils.api_run_planner import (
    FinalFitAction,
    SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION,
    build_composer_execution_plan,
    history_has_records,
    is_atomized_initial_assumption,
    plan_chunked_ensemble,
    plan_final_fit,
    plan_sampling_stage,
)
from fedot.core.repository.tasks import TaskTypesEnum


class _FakeHistory:
    def __init__(self, is_empty_value):
        self._is_empty_value = is_empty_value

    def is_empty(self):
        return self._is_empty_value


def test_is_atomized_initial_assumption_detects_atomized_pipeline():
    atomized = SimpleNamespace(descriptive_id='some_atomized_pipeline')
    regular = SimpleNamespace(descriptive_id='ordinary_pipeline')

    assert is_atomized_initial_assumption(atomized) is True
    assert is_atomized_initial_assumption(regular) is False
    assert is_atomized_initial_assumption(None) is False


def test_plan_sampling_stage_runs_when_sampling_config_present():
    plan = plan_sampling_stage(
        initial_assumption=None,
        sampling_config_present=True,
    )

    assert plan.should_run_sampling_stage is True
    assert plan.skip_metadata is None


def test_plan_sampling_stage_does_not_run_without_sampling_config():
    plan = plan_sampling_stage(
        initial_assumption=None,
        sampling_config_present=False,
    )

    assert plan.should_run_sampling_stage is False
    assert plan.skip_metadata is None


def test_plan_sampling_stage_skips_for_atomized_initial_assumption():
    atomized = SimpleNamespace(descriptive_id='my_atomized_pipeline')
    plan = plan_sampling_stage(
        initial_assumption=atomized,
        sampling_config_present=True,
    )

    assert plan.should_run_sampling_stage is False
    assert plan.skip_metadata == {'status': 'skipped',
                                  'reason': SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION}


def test_plan_sampling_stage_atomized_initial_assumption_skip_does_not_need_sampling_config():
    atomized = SimpleNamespace(descriptive_id='my_atomized_pipeline')
    plan = plan_sampling_stage(
        initial_assumption=atomized,
        sampling_config_present=False,
    )

    assert plan.should_run_sampling_stage is False
    assert plan.skip_metadata is None


def test_plan_chunked_ensemble_uses_holdout_for_supported_chunking_tasks():
    classification_plan = plan_chunked_ensemble(
        should_run_sampling_stage=True,
        strategy_kind='chunking',
        task_type=TaskTypesEnum.classification,
    )
    regression_plan = plan_chunked_ensemble(
        should_run_sampling_stage=True,
        strategy_kind='chunking',
        task_type=TaskTypesEnum.regression,
    )

    assert classification_plan.should_use_chunked_ensemble is True
    assert classification_plan.require_config().validation_size == 0.2
    assert classification_plan.train_split_ratio == 0.8
    assert classification_plan.should_select_class_representatives is True
    assert regression_plan.should_use_chunked_ensemble is True
    assert regression_plan.should_select_class_representatives is False


def test_plan_chunked_ensemble_uses_config_values():
    plan = plan_chunked_ensemble(
        should_run_sampling_stage=True,
        strategy_kind='chunking',
        task_type=TaskTypesEnum.regression,
        chunked_ensemble_config={
            'validation_size': 0.25,
            'validation_split_seed': 7,
            'ensemble_method': 'weighted',
        },
    )

    assert plan.should_use_chunked_ensemble is True
    assert plan.train_split_ratio == 0.75
    assert plan.validation_split_seed == 7
    assert plan.require_config().ensemble_method.value == 'weighted'


def test_plan_chunked_ensemble_skips_non_chunking_paths():
    plan = plan_chunked_ensemble(
        should_run_sampling_stage=True,
        strategy_kind='subset',
        task_type=TaskTypesEnum.classification,
    )
    no_sampling_plan = plan_chunked_ensemble(
        should_run_sampling_stage=False,
        strategy_kind='chunking',
        task_type=TaskTypesEnum.classification,
    )

    assert plan.should_use_chunked_ensemble is False
    assert plan.config is None
    assert no_sampling_plan.should_use_chunked_ensemble is False


def test_plan_final_fit_respects_history_and_pipeline_fit_state():
    assert history_has_records(None) is False
    assert history_has_records(_FakeHistory(is_empty_value=True)) is False
    assert history_has_records(_FakeHistory(is_empty_value=False)) is True

    assert plan_final_fit(None, pipeline_is_fitted=True).action is FinalFitAction.skip
    assert plan_final_fit(_FakeHistory(is_empty_value=False),
                          pipeline_is_fitted=True).action is FinalFitAction.fit_pipeline_on_full_data
    assert plan_final_fit(None, pipeline_is_fitted=False).action is FinalFitAction.fit_pipeline_on_full_data
    assert plan_final_fit(
        _FakeHistory(is_empty_value=False),
        pipeline_is_fitted=False,
        is_pipeline_ensemble=True,
    ).action is FinalFitAction.finalize_ensemble


def test_build_composer_execution_plan_is_typed_and_deterministic():
    plan = build_composer_execution_plan(
        with_tuning=True,
        have_time_for_composing=True,
        have_time_for_tuning=False,
        tuning_timeout_minutes=-5,
    )

    assert plan.should_compose is True
    assert plan.should_tune is False
    assert plan.tuning_timeout_minutes == 0.0
