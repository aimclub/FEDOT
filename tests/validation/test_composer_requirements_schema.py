import pytest

from fedot.validation.errors import FedotValidationError
from fedot.validation.schemas.composer_requirements import validate_cv_folds


def test_validate_cv_folds_accepts_valid_value():
    validate_cv_folds(5)


def test_validate_cv_folds_accepts_none():
    validate_cv_folds(None)


def test_validate_cv_folds_rejects_one():
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(1)


def test_validate_cv_folds_rejects_zero():
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(0)
