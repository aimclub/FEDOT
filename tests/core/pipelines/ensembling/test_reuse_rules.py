import pytest

from fedot.core.pipelines.ensembling.reuse_rules import plan_chunk_initial_population


@pytest.mark.unit
@pytest.mark.parametrize(
    ('reuse_enabled', 'previous_best_count', 'expected'),
    [
        (True, 2, True),
        (True, 0, False),
        (False, 2, False),
    ],
)
def test_plan_chunk_initial_population(
    reuse_enabled: bool,
    previous_best_count: int,
    expected: bool,
):
    plan = plan_chunk_initial_population(reuse_enabled, previous_best_count)

    assert plan.use_previous_best is expected
