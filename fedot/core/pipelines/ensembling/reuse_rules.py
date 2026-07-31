from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkInitialPopulationPlan:
    """Decision to seed a chunk search with the previous chunk's best pipelines."""

    use_previous_best: bool


def plan_chunk_initial_population(
    reuse_enabled: bool,
    previous_best_count: int,
) -> ChunkInitialPopulationPlan:
    """Resolve reuse without copying, fitting, or mutating pipeline objects."""
    return ChunkInitialPopulationPlan(
        use_previous_best=reuse_enabled and previous_best_count > 0,
    )
