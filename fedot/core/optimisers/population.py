"""FEDOT-side adapters over GOLEM's public population operators."""
from contextlib import contextmanager
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil, isfinite
import random
from threading import RLock
from typing import Callable, Optional, Protocol

import numpy as np
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, ObjectiveEvaluationDispatcher
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.utilities.utilities import determine_n_jobs
from joblib import Parallel, delayed


@dataclass(frozen=True)
class PopulationPlan:
    selected: tuple[int, ...]
    duplicates: tuple[int, ...]


def plan_unique_population(candidate_ids: tuple[str, ...], excluded: frozenset[str] = frozenset(),
                           preferred: frozenset[int] = frozenset()) -> PopulationPlan:
    """Choose one stable representative per identity without mutating input.

    A preferred occurrence replaces an earlier non-preferred occurrence while
    retaining the identity's original position in the output order. This lets
    dispatchers preserve an already evaluated duplicate instead of evaluating
    an equivalent pending individual again.
    """
    seen = set(excluded)
    selected, duplicates = [], []
    selected_position = {}
    for index, candidate_id in enumerate(candidate_ids):
        if candidate_id not in seen:
            seen.add(candidate_id)
            selected_position[candidate_id] = len(selected)
            selected.append(index)
            continue

        position = selected_position.get(candidate_id)
        if position is not None and index in preferred and selected[position] not in preferred:
            duplicates.append(selected[position])
            selected[position] = index
        else:
            duplicates.append(index)
    return PopulationPlan(tuple(selected), tuple(sorted(duplicates)))


def candidate_identity(individual) -> str:
    return individual.graph.descriptive_id


class ContractEvaluationDispatcher(MultiprocessingDispatcher):
    """GOLEM transport without its implicit last-chance evaluation retry."""

    def __init__(self, adapter, n_jobs=1, graph_cleanup_fn=None, delegate_evaluator=None):
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self.jobs = n_jobs

    def evaluate_population(self, individuals):
        pending, complete = self.split_individuals_to_evaluate(individuals)
        jobs = determine_n_jobs(self.jobs, self.logger)
        if jobs == 1:
            results = [self.evaluate_single(ind.graph, ind.uid) for ind in pending]
        else:
            from fedot.extensions import get_registered_extensions
            manifests = tuple(item.manifest for item in get_registered_extensions())
            results = Parallel(n_jobs=jobs, pre_dispatch='2*n_jobs')(
                delayed(_evaluate_with_extensions)(self.evaluate_single, ind.graph, ind.uid, manifests)
                for ind in pending)
        return self.apply_evaluation_results(pending, results) + complete

    def evaluate_with_cache(self, population):
        try:
            return super().evaluate_with_cache(population)
        finally:
            self.evaluation_cache.clear()


def _evaluate_with_extensions(evaluate, graph, uid, manifests):
    """Replay the caller's explicit FED-02 scope in a reused worker process."""
    from fedot.extensions import extension_scope

    def run():
        with extension_scope(*manifests):
            return evaluate(graph, uid)

    return Context().run(run)


class UniqueEvaluationDispatcher(ObjectiveEvaluationDispatcher):
    """Deduplicate before the GOLEM worker boundary, including initial population."""

    def __init__(self, delegate: ObjectiveEvaluationDispatcher):
        self.delegate = delegate
        self.last_plan = PopulationPlan((), ())

    def dispatch(self, objective, timer=None):
        evaluate = self.delegate.dispatch(objective, timer)

        def evaluate_unique(population):
            population = tuple(population)
            preferred = frozenset(index for index, individual in enumerate(population)
                                  if individual.fitness.valid)
            self.last_plan = plan_unique_population(
                tuple(map(candidate_identity, population)), preferred=preferred)
            if not self.last_plan.selected:
                return []
            return evaluate([population[i] for i in self.last_plan.selected])

        return evaluate_unique

    def set_graph_evaluation_callback(self, callback):
        self.delegate.set_graph_evaluation_callback(callback)


@dataclass(frozen=True)
class ReproductionPolicy:
    seed: int = 0
    max_attempts: int = 10
    required_valid_ratio: float = 1.0

    def __post_init__(self):
        if (isinstance(self.seed, bool) or not isinstance(self.seed, int)
                or isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int)
                or self.max_attempts < 1):
            raise ValueError('seed must be an integer; max_attempts must be positive')
        if isinstance(self.required_valid_ratio, bool) or not isfinite(self.required_valid_ratio) or not (
                0 < self.required_valid_ratio <= 1):
            raise ValueError('required_valid_ratio must be in (0, 1]')


@dataclass(frozen=True)
class ReproductionStep:
    attempt: int
    seed: int
    requested: int


def plan_reproduction(policy: ReproductionPolicy, generation: int, attempt: int,
                      target_size: int, collected: int, parent_count: int) -> ReproductionStep:
    values = (generation, attempt, target_size, collected, parent_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values) \
            or min(generation, collected, parent_count) < 0 or target_size < 1 \
            or not 1 <= attempt <= policy.max_attempts:
        raise ValueError('invalid reproduction state')
    seed_bytes = f'{policy.seed}:{generation}:{attempt}'.encode('ascii')
    seed = int.from_bytes(sha256(seed_bytes).digest()[:4], 'big')
    return ReproductionStep(attempt, seed, min(parent_count, max(0, target_size - collected)))


@dataclass(frozen=True)
class ReproductionReport:
    target_size: int
    required_size: int
    candidate_ids: tuple[str, ...]
    steps: tuple[ReproductionStep, ...]
    duplicate_count: int

    @property
    def complete(self) -> bool:
        return len(self.candidate_ids) >= self.required_size


class ReproductionHook(Protocol):
    def reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        ...


class UncontrolledReproducer(Protocol):
    def reproduce_uncontrolled(self, population: PopulationT, evaluator: EvaluationOperator,
                               pop_size: Optional[int] = None) -> PopulationT:
        ...


# GOLEM operators use Python/NumPy global RNGs. This adapter serializes only
# their short decision phase and restores both states, even if an operator fails.
_RNG_LOCK = RLock()


@contextmanager
def reproduction_random_state(seed):
    with _RNG_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class BoundedReproduction:
    """Use GOLEM selection/crossover/mutation; own only bounded collection policy."""

    def __init__(self, delegate: UncontrolledReproducer, policy: ReproductionPolicy,
                 target_size: Callable[[], int]):
        self.delegate = delegate
        self.policy = policy
        self.target_size = target_size
        self.generation = 0
        self.last_report: Optional[ReproductionReport] = None

    def reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        parents = tuple(population)
        target = self.target_size()
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError('population size must be a positive integer')
        required = ceil(target * self.policy.required_valid_ratio)
        collected, seen, steps, duplicate_count = {}, set(), [], 0
        generation = self.generation
        self.generation += 1
        for attempt in range(1, self.policy.max_attempts + 1):
            step = plan_reproduction(self.policy, generation, attempt, target, len(collected), len(parents))
            if not step.requested:
                break
            steps.append(step)
            # Generate only under RNG scope; expensive evaluation stays outside.
            with reproduction_random_state(step.seed):
                generated = tuple(self.delegate.reproduce_uncontrolled(
                    [replace(ind, graph=deepcopy(ind.graph), metadata=deepcopy(ind.metadata))
                     for ind in parents], lambda items: items, step.requested))
            ids = tuple(map(candidate_identity, generated))
            plan = plan_unique_population(ids, frozenset(seen))
            duplicate_count += len(plan.duplicates)
            seen.update(ids)
            unique = [generated[i] for i in plan.selected]
            evaluated = tuple(evaluator(unique)) if unique else ()
            # Fitting may normalize node parameters and change descriptive_id.
            # A worker's result belongs to its requested uid, not the rewritten graph.
            allowed = {generated[i].uid: ids[i] for i in plan.selected}
            for individual in evaluated:
                requested_id = allowed.get(individual.uid)
                if requested_id is not None and individual.fitness.valid:
                    collected.setdefault(requested_id, individual)
            if len(collected) >= required:
                break
        selected = tuple(collected)[:target]
        self.last_report = ReproductionReport(target, required, selected, tuple(steps), duplicate_count)
        if not self.last_report.complete:
            raise EvaluationAttemptsError(
                f'FEDOT reproduction incomplete: {len(selected)}/{required} unique valid candidates '
                f'after {len(steps)} attempts')
        return [collected[key] for key in selected]
