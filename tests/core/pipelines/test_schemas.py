import pytest

from fedot.validation.errors import FedotValidationError
from fedot.core.pipelines.schemas import (
    validate_cv_folds,
    validate_pipeline_is_fitted,
    validate_single_root_node,
)


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


def test_validate_pipeline_is_fitted_accepts_true():
    validate_pipeline_is_fitted(True)


def test_validate_pipeline_is_fitted_rejects_false():
    with pytest.raises(FedotValidationError, match='Pipeline is not fitted yet'):
        validate_pipeline_is_fitted(False)


def test_validate_single_root_node_accepts_single_root():
    validate_single_root_node(1)


def test_validate_single_root_node_rejects_multiple_roots():
    with pytest.raises(FedotValidationError, match='More than 1 root_nodes in pipeline'):
        validate_single_root_node(2)
