import pandas as pd

from fedot.industrial.tools.registry.model_registry_cleanup_rules import (
    build_registry_storage_cleanup_plan,
)
from fedot.industrial.tools.registry.model_registry_memory_policy_rules import (
    build_checkpoint_save_cleanup_plan,
    build_cleanup_efficiency_plan,
    build_memory_cleanup_plan,
)
from fedot.industrial.tools.registry.model_registry_rules import (
    build_registry_record_plan,
    build_registry_stage_mode_plan,
)
from fedot.industrial.tools.registry.registry_update_rules import (
    build_evaluator_metrics_update_plan,
    build_register_changes_plan,
)


def test_registry_save_planning_flow_preserves_stage_mode_and_checkpoint_policy():
    stage_mode_plan = build_registry_stage_mode_plan(
        stage='initial_fit',
        mode=None,
        latest_record={'mode': 'tensor_gpu_bridge'},
    )
    cleanup_plan = build_checkpoint_save_cleanup_plan(
        auto_cleanup=True,
        cuda_available=True,
        delete_model_after_save=False,
        parallel_workers=2,
    )
    record_plan = build_registry_record_plan(
        record_id='record_1',
        fedcore_id='fedcore_1',
        model_id='model_1',
        version='2026-04-09T12:00:00',
        checkpoint_path='checkpoints/model_1.pt',
        stage=stage_mode_plan.stage,
        mode=stage_mode_plan.mode,
    )

    assert stage_mode_plan.stage == 'before'
    assert stage_mode_plan.mode == 'tensor_gpu_bridge'
    assert cleanup_plan.cleanup_after_save is True
    assert cleanup_plan.cleanup_tier == 'standard'
    assert record_plan.stage == 'before'
    assert record_plan.mode == 'tensor_gpu_bridge'



def test_registry_cleanup_planning_flow_preserves_storage_and_memory_intent():
    storage_plan = build_registry_storage_cleanup_plan(['record', 'checkpoint_bytes', 'metrics'])
    memory_plan = build_memory_cleanup_plan(
        auto_cleanup=True,
        cuda_available=True,
        cleanup_iterations=3,
        comprehensive=True,
    )
    efficiency_plan = build_cleanup_efficiency_plan(5.0, 1.5)

    assert storage_plan.clear_checkpoint_bytes is True
    assert storage_plan.target_column == 'checkpoint_bytes'
    assert memory_plan.cleanup_tier == 'comprehensive'
    assert memory_plan.extra_cuda_cleanup_iterations == 3
    assert abs(efficiency_plan.memory_freed_gb - 3.5) < 1e-9
    assert abs(efficiency_plan.efficiency_percent - 70.0) < 1e-9



def test_registry_update_planning_flow_routes_existing_and_extracts_last_generation_metrics():
    register_changes_plan = build_register_changes_plan(
        existing_record={'model': 'model_1'},
        stage='after_fit',
        mode='input_cpu',
    )
    metrics_plan = build_evaluator_metrics_update_plan(
        pd.DataFrame([
            {'generation': 0, 'metric_0': 0.1},
            {'generation': 1, 'metric_0': 0.4, 'metric_1': 0.8},
        ])
    )

    assert register_changes_plan.should_register_new is False
    assert register_changes_plan.should_save_changes is True
    assert metrics_plan.should_update is True
    assert metrics_plan.metrics == {'metric_0': 0.4, 'metric_1': 0.8}