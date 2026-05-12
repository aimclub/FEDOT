from types import SimpleNamespace

from fedot.api.api_utils.api_run_planner import (
    SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION,
    SKIP_REASON_PREDEFINED_MODEL,
    build_composer_execution_plan,
    history_has_records,
    is_atomized_initial_assumption,
    plan_final_fit,
    plan_sampling_stage,
)


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


def test_plan_sampling_stage_skips_for_explicit_predefined_model():
    plan = plan_sampling_stage(
        requested_predefined_model='rf',
        initial_assumption=None,
        sampling_config_present=True,
    )

    assert plan.resolved_predefined_model == 'rf'
    assert plan.should_run_sampling_stage is False
    assert plan.skip_metadata == {
        'status': 'skipped', 'reason': SKIP_REASON_PREDEFINED_MODEL}


def test_plan_sampling_stage_skips_for_atomized_initial_assumption():
    atomized = SimpleNamespace(descriptive_id='my_atomized_pipeline')
    plan = plan_sampling_stage(
        requested_predefined_model=None,
        initial_assumption=atomized,
        sampling_config_present=True,
    )

    assert plan.resolved_predefined_model is atomized
    assert plan.should_run_sampling_stage is False
    assert plan.skip_metadata == {'status': 'skipped',
                                  'reason': SKIP_REASON_ATOMIZED_INITIAL_ASSUMPTION}


def test_plan_sampling_stage_runs_only_when_sampling_config_present():
    plan = plan_sampling_stage(
        requested_predefined_model=None,
        initial_assumption=None,
        sampling_config_present=True,
    )
    no_sampling_plan = plan_sampling_stage(
        requested_predefined_model=None,
        initial_assumption=None,
        sampling_config_present=False,
    )

    assert plan.should_run_sampling_stage is True
    assert plan.skip_metadata is None
    assert no_sampling_plan.should_run_sampling_stage is False


def test_plan_final_fit_respects_history_and_pipeline_fit_state():
    assert history_has_records(None) is False
    assert history_has_records(_FakeHistory(is_empty_value=True)) is False
    assert history_has_records(_FakeHistory(is_empty_value=False)) is True

    assert plan_final_fit(
        None, pipeline_is_fitted=True).should_train_on_full_dataset is False
    assert plan_final_fit(_FakeHistory(is_empty_value=False),
                          pipeline_is_fitted=True).should_train_on_full_dataset is True
    assert plan_final_fit(
        None, pipeline_is_fitted=False).should_train_on_full_dataset is True


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
