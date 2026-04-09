import pandas as pd

from fedot.industrial.tools.registry.registry_update_rules import (
    EvaluatorMetricsUpdatePlan,
    RegisterChangesPlan,
    build_evaluator_metrics_update_plan,
    build_register_changes_plan,
)


def test_build_register_changes_plan_routes_between_new_registration_and_save_changes():
    no_existing = build_register_changes_plan(
        existing_record=None,
        stage='before_fit',
        mode='tensor_gpu_bridge',
    )
    assert isinstance(no_existing, RegisterChangesPlan)
    assert no_existing.should_register_new is True
    assert no_existing.should_save_changes is False
    assert no_existing.stage == 'before_fit'
    assert no_existing.mode == 'tensor_gpu_bridge'

    existing = build_register_changes_plan(
        existing_record={'model': 'model_1'},
        stage='after_fit',
        mode='input_cpu',
    )
    assert existing.should_register_new is False
    assert existing.should_save_changes is True



def test_build_evaluator_metrics_update_plan_uses_last_generation_without_generation_key():
    metrics_df = pd.DataFrame([
        {'generation': 0, 'metric_0': 0.1},
        {'generation': 1, 'metric_0': 0.3, 'metric_1': 0.7},
    ])

    plan = build_evaluator_metrics_update_plan(metrics_df)
    assert isinstance(plan, EvaluatorMetricsUpdatePlan)
    assert plan.should_update is True
    assert plan.metrics == {'metric_0': 0.3, 'metric_1': 0.7}



def test_build_evaluator_metrics_update_plan_skips_empty_metrics_frames():
    empty_plan = build_evaluator_metrics_update_plan(pd.DataFrame())
    assert empty_plan.should_update is False
    assert empty_plan.metrics == {}

    generation_only_plan = build_evaluator_metrics_update_plan(pd.DataFrame([{'generation': 3}]))
    assert generation_only_plan.should_update is False
    assert generation_only_plan.metrics == {}