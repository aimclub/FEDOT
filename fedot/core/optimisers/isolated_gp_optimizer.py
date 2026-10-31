"""GP optimizer using per-candidate process isolation for population evaluation."""

from __future__ import annotations

from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer

from fedot.core.optimisers.isolated_candidate_dispatcher import IsolatedCandidateDispatcher


class IsolatedEvoGraphOptimizer(EvoGraphOptimizer):
    """Drop-in FEDOT optimizer that preserves GA policy and isolates candidate fits."""

    def __init__(self, *args, candidate_startup_timeout_seconds: float = 60.0, **kwargs):
        super().__init__(*args, **kwargs)
        dispatcher = self.eval_dispatcher
        if isinstance(dispatcher, MultiprocessingDispatcher) \
                and not isinstance(dispatcher, IsolatedCandidateDispatcher):
            self.eval_dispatcher = IsolatedCandidateDispatcher(
                adapter=dispatcher._adapter,
                n_jobs=dispatcher._n_jobs,
                graph_cleanup_fn=dispatcher._cleanup,
                delegate_evaluator=dispatcher._delegate_evaluator,
                startup_timeout_seconds=candidate_startup_timeout_seconds,
            )
