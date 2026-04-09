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