"""Pure, immutable decisions for complete cross-validation evaluation."""
from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite
from typing import Optional, Protocol, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class FailureKind(str, Enum):
    DATA = 'data'
    VALIDATION = 'validation'
    FIT = 'fit'
    METRIC = 'metric'
    TIMEOUT = 'timeout'
    TRANSIENT = 'transient'
    CLEANUP = 'cleanup'
    CACHE = 'cache'


@dataclass(frozen=True)
class EvaluationFailure:
    kind: FailureKind
    message: str
    exception_type: str = ''


class RetryableEvaluationError(RuntimeError):
    """Explicit opt-in for a transient evaluator/backend failure."""


@dataclass(frozen=True)
class RetryPolicy:
    """Maximum attempts includes the first call. No implicit timeout retry."""
    max_attempts: int = 1
    retryable: tuple[FailureKind, ...] = (FailureKind.TRANSIENT,)

    def __post_init__(self):
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError('max_attempts must be a positive integer')
        if not isinstance(self.retryable, tuple) or any(
                not isinstance(kind, FailureKind) for kind in self.retryable):
            raise TypeError('retryable must be a tuple of FailureKind values')
        if set(self.retryable) - {FailureKind.TRANSIENT, FailureKind.TIMEOUT}:
            raise ValueError('only explicitly transient failures and timeouts may be retried')


def should_retry(policy: RetryPolicy, failure: EvaluationFailure, attempt: int) -> bool:
    return 1 <= attempt < policy.max_attempts and failure.kind in policy.retryable


@dataclass(frozen=True)
class AttemptRecord:
    fold_id: int
    attempt: int
    failure: Optional[EvaluationFailure] = None
    cleanup_failure: Optional[EvaluationFailure] = None


@dataclass(frozen=True)
class FoldRecord:
    fold_id: int
    metrics: tuple[float, ...]
    attempts: int
    cache_key: str

    def __post_init__(self):
        invalid_type = (isinstance(self.fold_id, bool) or not isinstance(self.fold_id, int)
                        or isinstance(self.attempts, bool) or not isinstance(self.attempts, int))
        if invalid_type or self.fold_id < 0 or self.attempts < 1:
            raise ValueError('fold_id and attempts are out of range')
        if not isinstance(self.metrics, tuple) or not self.metrics or not all(map(isfinite, self.metrics)):
            raise ValueError('fold metrics must be a nonempty finite tuple')


@dataclass(frozen=True)
class EvaluationComplete:
    candidate_id: str
    expected_folds: int
    folds: tuple[FoldRecord, ...]
    attempts: tuple[AttemptRecord, ...]

    def __post_init__(self):
        if not isinstance(self.folds, tuple) or not isinstance(self.attempts, tuple):
            raise TypeError('folds and attempts must be immutable tuples')
        if isinstance(self.expected_folds, bool) or not isinstance(self.expected_folds, int) \
                or self.expected_folds < 1 or tuple(
                f.fold_id for f in self.folds) != tuple(range(self.expected_folds)):
            raise ValueError('a complete evaluation must contain every fold exactly once, in order')
        if len({len(f.metrics) for f in self.folds}) != 1:
            raise ValueError('all folds must have the same metric dimension')

    @property
    def metrics(self) -> tuple[float, ...]:
        return tuple(fsum(values) / self.expected_folds for values in zip(*(f.metrics for f in self.folds)))


@dataclass(frozen=True)
class EvaluationIncomplete:
    candidate_id: str
    expected_folds: int
    folds: tuple[FoldRecord, ...]
    attempts: tuple[AttemptRecord, ...]
    failure: EvaluationFailure

    def __post_init__(self):
        if not isinstance(self.folds, tuple) or not isinstance(self.attempts, tuple):
            raise TypeError('folds and attempts must be immutable tuples')
        if isinstance(self.expected_folds, bool) or not isinstance(self.expected_folds, int) \
                or self.expected_folds < 0 \
                or tuple(f.fold_id for f in self.folds) != tuple(range(len(self.folds))):
            raise ValueError('incomplete evaluation must contain an ordered successful prefix')


@dataclass(frozen=True)
class EvaluationReused:
    """Reference to a prior success, not another successful evaluation."""
    original: EvaluationComplete


EvaluationOutcome = Union[EvaluationComplete, EvaluationIncomplete, EvaluationReused]


@dataclass(frozen=True)
class ValidationResult:
    violations: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.violations, tuple) or any(not isinstance(x, str) or not x for x in self.violations):
            raise TypeError('violations must be a tuple of nonempty strings')

    @property
    def valid(self) -> bool:
        return not self.violations


class PipelineValidator(Protocol):
    def __call__(self, pipeline: 'Pipeline') -> ValidationResult:
        """Inspect, but do not mutate, the candidate."""
        ...


def validate_pipeline(pipeline: 'Pipeline') -> ValidationResult:
    """Minimal evaluation boundary; task-specific rules remain in GOLEM."""
    return ValidationResult(()) if pipeline.nodes else ValidationResult(('pipeline has no nodes',))


def failure_from_exception(error: Exception, phase: FailureKind) -> EvaluationFailure:
    if isinstance(error, RetryableEvaluationError):
        kind = FailureKind.TRANSIENT
    elif isinstance(error, TimeoutError):
        kind = FailureKind.TIMEOUT
    else:
        kind = phase
    return EvaluationFailure(kind, str(error), type(error).__name__)
