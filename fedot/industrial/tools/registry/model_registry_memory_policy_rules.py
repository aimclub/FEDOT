"""Pure memory policy rules for the model registry shell."""

from dataclasses import dataclass
from typing import Literal, Optional

CleanupTier = Literal['none', 'standard', 'force', 'comprehensive']


@dataclass(frozen=True)
class MemoryStatsPlan:
    enabled: bool
    context: str


@dataclass(frozen=True)
class CheckpointSaveCleanupPlan:
    cleanup_after_save: bool
    cleanup_tier: CleanupTier
    should_log_before_save: bool
    should_log_after_save: bool


@dataclass(frozen=True)
class MemoryCleanupPlan:
    cleanup_tier: CleanupTier
    run_checkpoint_manager_cleanup: bool
    extra_cuda_cleanup_iterations: int
    extra_gc_iterations: int


@dataclass(frozen=True)
class CleanupEfficiencyPlan:
    memory_freed_gb: float
    efficiency_percent: Optional[float]


def resolve_parallel_worker_count(parallel_workers: Optional[int] = None) -> int:
    if parallel_workers is None:
        return 1
    try:
        normalized = int(parallel_workers)
    except (TypeError, ValueError):
        return 1
    return max(1, normalized)


def build_memory_stats_plan(auto_cleanup: bool,
                            cuda_available: bool,
                            context: str) -> MemoryStatsPlan:
    return MemoryStatsPlan(
        enabled=bool(auto_cleanup and cuda_available),
        context=context,
    )


def build_checkpoint_save_cleanup_plan(auto_cleanup: bool,
                                       cuda_available: bool,
                                       delete_model_after_save: bool,
                                       cleanup_after_save: Optional[bool] = None,
                                       parallel_workers: Optional[int] = None) -> CheckpointSaveCleanupPlan:
    worker_count = resolve_parallel_worker_count(parallel_workers)
    resolved_cleanup = bool((auto_cleanup if cleanup_after_save is None else cleanup_after_save) and cuda_available)

    if not resolved_cleanup:
        cleanup_tier: CleanupTier = 'none'
    elif worker_count > 1:
        cleanup_tier = 'standard'
    else:
        cleanup_tier = 'standard'

    should_log = bool(auto_cleanup and cuda_available)
    return CheckpointSaveCleanupPlan(
        cleanup_after_save=resolved_cleanup,
        cleanup_tier=cleanup_tier,
        should_log_before_save=should_log,
        should_log_after_save=should_log and (resolved_cleanup or delete_model_after_save),
    )


def build_memory_cleanup_plan(auto_cleanup: bool,
                              cuda_available: bool,
                              cleanup_iterations: int,
                              force: bool = False,
                              comprehensive: bool = False,
                              parallel_workers: Optional[int] = None) -> MemoryCleanupPlan:
    _ = resolve_parallel_worker_count(parallel_workers)

    if force:
        return MemoryCleanupPlan(
            cleanup_tier='force',
            run_checkpoint_manager_cleanup=True,
            extra_cuda_cleanup_iterations=0,
            extra_gc_iterations=0,
        )

    if comprehensive:
        return MemoryCleanupPlan(
            cleanup_tier='comprehensive',
            run_checkpoint_manager_cleanup=True,
            extra_cuda_cleanup_iterations=cleanup_iterations if cuda_available else 0,
            extra_gc_iterations=cleanup_iterations,
        )

    if auto_cleanup and cuda_available:
        return MemoryCleanupPlan(
            cleanup_tier='standard',
            run_checkpoint_manager_cleanup=True,
            extra_cuda_cleanup_iterations=1,
            extra_gc_iterations=1,
        )

    return MemoryCleanupPlan(
        cleanup_tier='none',
        run_checkpoint_manager_cleanup=False,
        extra_cuda_cleanup_iterations=0,
        extra_gc_iterations=0,
    )


def build_cleanup_efficiency_plan(memory_before_gb: float,
                                  memory_after_gb: float) -> CleanupEfficiencyPlan:
    memory_freed = float(memory_before_gb) - float(memory_after_gb)
    if memory_before_gb > 0:
        efficiency = (memory_freed / float(memory_before_gb)) * 100
    else:
        efficiency = None

    return CleanupEfficiencyPlan(
        memory_freed_gb=memory_freed,
        efficiency_percent=efficiency,
    )
