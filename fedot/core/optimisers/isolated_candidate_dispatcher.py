"""Multiprocessing dispatcher with a killable process boundary per candidate."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from functools import partial
from typing import Optional

from golem.core.log import Log
from golem.core.optimisers.genetic.evaluation import (
    EvalResultsList,
    MultiprocessingDispatcher,
)
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.core.optimisers.timer import Timer
from golem.utilities.memory import MemoryAnalytics

from fedot.core.optimisers.candidate_isolation import (
    CandidateCleanupError,
    run_candidates_isolated,
)


class IsolatedCandidateDispatcher(MultiprocessingDispatcher):
    """Evaluate population candidates independently and retain completed siblings.

    ``PipelineObjectiveEvaluate`` supplies a cumulative candidate budget. This
    class applies that same budget as a killable process deadline, additionally
    bounded by the optimizer's remaining shared search timer.
    """

    def __init__(self, *args, startup_timeout_seconds: float = 60.0, **kwargs):
        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be finite and positive")
        self._startup_timeout_seconds = startup_timeout_seconds
        super().__init__(*args, **kwargs)

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        if not individuals_to_evaluate:
            return individuals_to_skip

        shared_deadline = self._shared_search_deadline_monotonic()
        timeout_seconds = self._candidate_timeout_seconds()
        if timeout_seconds <= 0:
            raise EvaluationAttemptsError("no candidate produced a valid fitness: FEDOT search budget expired")

        eval_func = partial(self.evaluate_single, logs_initializer=Log().get_parameters())
        requests = [
            (individual.uid, (individual.graph, individual.uid), {})
            for individual in individuals_to_evaluate
        ]
        try:
            outcomes = run_candidates_isolated(
                evaluator=eval_func,
                requests=requests,
                n_jobs=self._n_jobs,
                timeout_seconds=timeout_seconds,
                shared_deadline_monotonic=shared_deadline,
                startup_timeout_seconds=self._startup_timeout_seconds,
                event_callback=self._log_candidate_event,
            )
        except CandidateCleanupError as error:
            self.logger.error(
                "candidate_evaluation aborted cleanup_error completed_outcomes=%s "
                "unresolved_candidate_pids=%s",
                [
                    {"uid": outcome.uid, "status": outcome.status,
                     "pid": outcome.pid,
                     "fitness_values": getattr(getattr(outcome.value, "fitness", None), "values", None)}
                    for outcome in error.outcomes
                ],
                list(error.unresolved_candidate_pids),
            )
            raise

        evaluation_results: EvalResultsList = []
        for outcome in outcomes:
            if outcome.status == "ok" and outcome.value is not None:
                evaluation_results.append(outcome.value)

        evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        successful = evaluated + individuals_to_skip
        self.population_evaluation_info(
            evaluated_pop_size=len(successful),
            pop_size=len(individuals),
        )
        if not successful:
            summary = ", ".join(f"{outcome.uid}:{outcome.status}" for outcome in outcomes)
            raise EvaluationAttemptsError(
                f"no candidate produced a valid fitness; candidate outcomes: {summary}"
            )

        MemoryAnalytics.log(
            self.logger,
            additional_info="isolated candidate evaluation of population",
        )
        return successful

    def _candidate_timeout_seconds(self) -> float:
        budget_seconds = self._candidate_fit_budget_seconds()
        timer: Optional[Timer] = self.timer
        if timer is not None and timer.timeout is not None:
            remaining = timer.timeout.total_seconds() - timer.seconds_from_start
            budget_seconds = remaining if budget_seconds is None else min(budget_seconds, remaining)

        if budget_seconds is None or not math.isfinite(budget_seconds):
            raise ValueError(
                "isolated candidate evaluation requires a finite candidate or search budget"
            )
        return max(0.0, budget_seconds)

    def _candidate_fit_budget_seconds(self) -> Optional[float]:
        objective_owner = getattr(self._objective_eval, "__self__", None) or self._objective_eval
        # Guard-era evaluators store the whole-candidate budget. Vanilla stores
        # the per-fit limit and applies it once per CV fold.
        whole_candidate_budget = getattr(objective_owner, "_evaluation_time_constraint", None)
        per_fit_budget = getattr(objective_owner, "_time_constraint", None)
        if whole_candidate_budget is not None:
            budget = whole_candidate_budget
            fold_count = 1
        else:
            budget = per_fit_budget
            fold_count = self._cv_fold_count(objective_owner)
        if budget is None:
            return None
        if not isinstance(budget, timedelta):
            raise ValueError("candidate evaluation budget must be a timedelta")
        return budget.total_seconds() * fold_count

    @staticmethod
    def _cv_fold_count(objective_owner) -> int:
        producer = getattr(objective_owner, "_data_producer", None)
        raw_folds = getattr(producer, "keywords", {}).get("cv_folds")
        if isinstance(raw_folds, int) and not isinstance(raw_folds, bool) and raw_folds >= 2:
            return raw_folds
        return 1

    def _shared_search_deadline_monotonic(self) -> Optional[float]:
        timer: Optional[Timer] = self.timer
        if timer is None or timer.timeout is None:
            return None
        remaining = timer.timeout.total_seconds() - timer.seconds_from_start
        return time.monotonic() + max(0.0, remaining)

    def _log_candidate_event(self, event: dict) -> None:
        self.logger.info(
            "candidate_evaluation event=%s uid=%s pid=%s elapsed_seconds=%.3f "
            "serialization_seconds=%.3f startup_seconds=%.3f payload_bytes=%s "
            "cleanup_confirmed=%s error=%s",
            event.get("event"),
            event.get("uid"),
            event.get("pid"),
            float(event.get("elapsed_seconds") or 0.0),
            float(event.get("serialization_seconds") or 0.0),
            float(event.get("startup_seconds") or 0.0),
            event.get("payload_bytes"),
            event.get("cleanup_confirmed"),
            str(event.get("error") or "").replace("\n", " | ")[:3000],
        )
