from fedot.industrial.tools.registry.model_registry_memory_policy_rules import (
    build_checkpoint_save_cleanup_plan,
    build_cleanup_efficiency_plan,
    build_memory_cleanup_plan,
    build_memory_stats_plan,
    resolve_parallel_worker_count,
)


def test_resolve_parallel_worker_count_normalizes_invalid_values_to_one():
    assert resolve_parallel_worker_count() == 1
    assert resolve_parallel_worker_count(0) == 1
    assert resolve_parallel_worker_count(-3) == 1
    assert resolve_parallel_worker_count('bad') == 1
    assert resolve_parallel_worker_count(4) == 4


def test_build_memory_stats_plan_only_enables_logging_for_cuda_auto_cleanup():
    assert build_memory_stats_plan(auto_cleanup=False, cuda_available=True, context='fit').enabled is False
    assert build_memory_stats_plan(auto_cleanup=True, cuda_available=False, context='fit').enabled is False
    plan = build_memory_stats_plan(auto_cleanup=True, cuda_available=True, context='fit')
    assert plan.enabled is True
    assert plan.context == 'fit'


def test_build_checkpoint_save_cleanup_plan_resolves_cleanup_and_logging_flags():
    disabled = build_checkpoint_save_cleanup_plan(
        auto_cleanup=False,
        cuda_available=True,
        delete_model_after_save=False,
    )
    assert disabled.cleanup_after_save is False
    assert disabled.cleanup_tier == 'none'
    assert disabled.should_log_before_save is False

    enabled = build_checkpoint_save_cleanup_plan(
        auto_cleanup=True,
        cuda_available=True,
        delete_model_after_save=True,
        parallel_workers=2,
    )
    assert enabled.cleanup_after_save is True
    assert enabled.cleanup_tier == 'standard'
    assert enabled.should_log_before_save is True
    assert enabled.should_log_after_save is True


def test_build_memory_cleanup_plan_distinguishes_force_and_comprehensive_paths():
    force_plan = build_memory_cleanup_plan(
        auto_cleanup=True,
        cuda_available=True,
        cleanup_iterations=3,
        force=True,
    )
    assert force_plan.cleanup_tier == 'force'
    assert force_plan.run_checkpoint_manager_cleanup is True
    assert force_plan.extra_cuda_cleanup_iterations == 0
    assert force_plan.extra_gc_iterations == 0

    comprehensive_plan = build_memory_cleanup_plan(
        auto_cleanup=True,
        cuda_available=True,
        cleanup_iterations=3,
        comprehensive=True,
    )
    assert comprehensive_plan.cleanup_tier == 'comprehensive'
    assert comprehensive_plan.extra_cuda_cleanup_iterations == 3
    assert comprehensive_plan.extra_gc_iterations == 3


def test_build_memory_cleanup_plan_returns_none_tier_when_cleanup_disabled():
    plan = build_memory_cleanup_plan(
        auto_cleanup=False,
        cuda_available=False,
        cleanup_iterations=3,
    )
    assert plan.cleanup_tier == 'none'
    assert plan.run_checkpoint_manager_cleanup is False
    assert plan.extra_cuda_cleanup_iterations == 0
    assert plan.extra_gc_iterations == 0


def test_build_cleanup_efficiency_plan_handles_zero_baseline():
    plan = build_cleanup_efficiency_plan(4.0, 1.5)
    assert abs(plan.memory_freed_gb - 2.5) < 1e-9
    assert abs(plan.efficiency_percent - 62.5) < 1e-9

    zero_plan = build_cleanup_efficiency_plan(0.0, 0.0)
    assert zero_plan.memory_freed_gb == 0.0
    assert zero_plan.efficiency_percent is None