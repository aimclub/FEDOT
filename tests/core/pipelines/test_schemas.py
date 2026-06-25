import pytest

from fedot.validation.errors import FedotValidationError
from fedot.core.pipelines.schemas import validate_cv_folds


def test_validate_cv_folds_accepts_valid_value():
    """A positive integer >= 2 must pass without raising."""
    validate_cv_folds(5)


def test_validate_cv_folds_accepts_none():
    """None (no cross-validation) must be accepted.

    Desired behavior: when the user does not request cross-validation,
    ``cv_folds=None`` is a valid config that the composer interprets as
    "use a simple train/test split instead".
    """
    validate_cv_folds(None)


def test_validate_cv_folds_rejects_one():
    """1 fold is degenerate (no train/test split) and must be rejected.

    Desired behavior: KFold with k=1 would evaluate on the same data used for
    training, which is meaningless. The validator must raise with 'must be 2
    or more' so the user understands the minimum.
    """
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(1)


def test_validate_cv_folds_rejects_zero():
    """Zero folds is nonsensical and must be rejected (same message as k=1)."""
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(0)
