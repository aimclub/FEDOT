from fedot.core.repository.tasks import TsForecastingParams
from fedot.industrial.api.utils.api_init_rules import (
    build_api_manager_state_plan,
    build_industrial_context_plan,
    build_learning_loss_plan,
    resolve_initial_assumption_problem,
)


def test_build_industrial_context_plan_detects_tabular_and_forecasting_context():
    plan = build_industrial_context_plan(
        problem='ts_forecasting',
        strategy='tabular_strategy',
        task_params={'forecast_length': 12},
        regression_tasks=['ts_forecasting', 'regression'],
    )

    assert plan.strategy_name == 'tabular_strategy'
    assert plan.is_default_fedot_context is True
    assert plan.is_regression_task_context is True
    assert plan.is_forecasting_context is True
    assert isinstance(plan.normalized_task_params, TsForecastingParams)
    assert plan.normalized_task_params.forecast_length == 12


def test_resolve_initial_assumption_problem_appends_strategy_only_for_default_fedot_context():
    assert resolve_initial_assumption_problem('classification', 'tabular', True) == 'classification_tabular'
    assert resolve_initial_assumption_problem('classification', 'default', False) == 'classification'


def test_build_learning_loss_plan_supports_dict_and_callable():
    callable_plan = build_learning_loss_plan(lambda x: x)
    dict_plan = build_learning_loss_plan({'quality_loss': 'f1', 'computational_loss': 'time'})

    assert callable(callable_plan.quality_loss)
    assert callable_plan.computational_loss is None
    assert dict_plan.quality_loss == 'f1'
    assert dict_plan.computational_loss == 'time'
    assert dict_plan.structural_loss is None


def test_build_api_manager_state_plan_sets_expected_defaults():
    state = build_api_manager_state_plan()

    assert state.solver is None
    assert state.predicted_labels is None
    assert state.predicted_probs is None
    assert state.predict_data is None
    assert state.dask_client is None
    assert state.dask_cluster is None
    assert state.target_encoder is None
    assert state.is_finetuned is False
